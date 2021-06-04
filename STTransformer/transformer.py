"""
Author: Jiacheng WU
The idea is mainly from Transformer (https://arxiv.org/abs/1706.03762),
and TimeSformer (https://arxiv.org/abs/2102.05095).
We novelly modified the TimeSformer for spatio-temporal prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class SpaceTimeTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        d_model = configs.d_model
        self.device = configs.device
        self.input_dim = configs.input_dim * configs.patch_size[0] * configs.patch_size[1]
        self.src_emb = input_embedding(self.input_dim, d_model, configs.emb_spatial_size,
                                       configs.input_length, self.device)
        self.tgt_emb = input_embedding(self.input_dim, d_model, configs.emb_spatial_size,
                                       configs.output_length, self.device)

        encoder_layer = EncoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout)
        decoder_layer = DecoderLayer(d_model, configs.nheads, configs.dim_feedforward, configs.dropout)
        self.encoder = Encoder(encoder_layer, num_layers=configs.num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_layers=configs.num_decoder_layers)
        self.linear_output = nn.Linear(d_model, self.input_dim)

    def forward(self, src, tgt, src_mask=None, memory_mask=None,
                train=True, ssr_ratio=0):
        """
        Args:
            src: (N, T_src, C, H, W)
            tgt: (N, T_tgt, C, H, W), T_tgt is 1 during test
            src_mask: (T_src, T_src)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, T_src)
        Returns:
            sst_pred: (N, T_tgt, H, W)
            nino_pred: (N, 24)
        """
        memory = self.encode(src, src_mask)
        if train:
            with torch.no_grad():
                # tgt = torch.cat([src[:, -1:], tgt[:, :-1]], dim=1)  # (N, T_tgt, C, H, W)
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sst_pred = self.decode(torch.cat([src[:, -1:], tgt[:, :-1]], dim=1),
                                       memory, tgt_mask, memory_mask)  # (N, T_tgt, C, H, W)

            if ssr_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(ssr_ratio *
                        torch.ones(tgt.size(0), tgt.size(1) - 1, 1, 1, 1)).to(self.device)
            else:
                teacher_forcing_mask = 0
            tgt = teacher_forcing_mask * tgt[:, :-1] + (1 - teacher_forcing_mask) * sst_pred[:, :-1]
            tgt = torch.cat([src[:, -1:], tgt], dim=1)
            sst_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
        else:
            if tgt is None:
                tgt = src[:, -1:]  # use last src as the input during test
            else:
                assert tgt.size(1) == 1
            for t in range(self.configs.output_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                sst_pred = self.decode(tgt, memory, tgt_mask, memory_mask)
                tgt = torch.cat([tgt, sst_pred[:, -1:]], dim=1)

        sst_pred = sst_pred[:, :, 0]  # (N, T_tgt, H, W)
        nino_pred = sst_pred[:, :, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)

        return sst_pred, nino_pred

    def encode(self, src, src_mask):
        """
        Args:
            src: (N, T_src, C, H, W)
            src_mask: (T_src, T_src)
        Returns:
            memory: (N, S, T_src, D)
        """
        T = src.size(1)
        src = unfold_StackOverChannel(src, self.configs.patch_size)  # (N, T_src, C_, H_, W_)
        src = src.reshape(src.size(0), T, self.input_dim, -1).permute(0, 3, 1, 2)  # (N, S, T_src, C_)
        src = self.src_emb(src)  # (N, S, T_src, D)
        memory = self.encoder(src, src_mask)  # (N, S, T_src, D)
        return memory

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        """
        Args:
            tgt: (N, T_tgt, C, H, W)
            memory: (N, S, T_src, D)
            tgt_mask: (T_tgt, T_tgt)
            memory_mask: (T_tgt, T_src)
        Returns:
            (N, T_tgt, C, H, W)
        """
        H, W = tgt.size()[-2:]
        T = tgt.size(1)
        tgt = unfold_StackOverChannel(tgt, self.configs.patch_size)  # (N, T_tgt, C_, H_, W_)
        tgt = tgt.reshape(tgt.size(0), T, self.input_dim, -1).permute(0, 3, 1, 2)  # (N, S, T_tgt, C_)
        tgt = self.tgt_emb(tgt)  # (N, S, T_tgt, D)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        output = self.linear_output(output).permute(0, 2, 3, 1)  # (N, T_tgt, C_, S)

        # (N, T_tgt, C_, H_, W_)
        output = output.reshape(tgt.size(0), T, self.input_dim,
                                H // self.configs.patch_size[0], W // self.configs.patch_size[1])
        # (N, T_tgt, C, H, W)
        output = fold_tensor(output, output_size=(H, W), kernel_size=self.configs.patch_size)
        return output

    def generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf')
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.configs.device)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask, memory_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_space_time_attn(self, query, key, value, mask):
        """
        Apply space and time attention sequentially
        Args:
            query (N, S, T, D)
            key (N, S, T, D)
            value (N, S, T, D)
        Returns:
            (N, S, T, D)
        """
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m, mask)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.encoder_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.time_attn = MultiHeadedAttention(d_model, nheads, TimeAttention, dropout)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SpaceAttention, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_space_time_attn(self, query, key, value, mask=None):
        """
        Apply space and time attention sequentially
        Args:
            query (N, S, T_q, D)
            key (N, S, T, D)
            value (N, S, T, D)
        Returns:
            (N, S, T_q, D)
        """
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m, mask)
    '''
    def joint_space_time_attention(self, query, key, value, mask):
        """
        not applicable in the decoder when the memory is attened,
        as the time length of query is not the same as T
        Args:
            query (N, S, T_q, D) -> (N, 1, S*T_q, D)
            key (N, S, T, D)
            value (N, S, T, D)
        Returns:
            (N, S, T_q, D)
        """
        nbatches, nspace, ntime = query.size()[:3]
        query = query.flatten(1, 2).unsqueeze(1)
        key = key.flatten(1, 2).unsqueeze(1)
        value = value.flatten(1, 2).unsqueeze(1)
        output = self.encoder_attn(query, key, value, mask)  # (N, 1, S*T_q, D)
        output = output.reshape(nbatches, nspace, ntime, -1)  # (N, S, T_q, D)
        return output
    '''
    def forward(self, x, memory, tgt_mask, memory_mask):
        x = self.sublayer[0](x, lambda x: self.divided_space_time_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.encoder_attn(x, memory, memory, memory_mask))
        return self.sublayer[2](x, self.feed_forward)


def unfold_StackOverChannel(img, kernel_size):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, *, C*H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, *, C, n0, n1, k0*k1)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * kernel_size[0] * kernel_size[1]
    return pt


def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor of size (N, *, C*k_h*k_w, n_h, n_w)
        output_size of size(H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class input_embedding(nn.Module):
    def __init__(self, input_dim, d_model, emb_spatial_size, max_len, device):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe_time = pe[None, None].to(device)  # (1, 1, T, D)
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
        self.emb_space = nn.Embedding(emb_spatial_size, d_model)
        self.linear = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Add temporal encoding and learnable spatial embedding to the input (after patch)
        Args:
            input x of size (N, S, T, in_channels)
        Returns:
            embedded input (N, S, T, D)
        """
        assert len(x.size()) == 4 
        embedded_space = self.emb_space(self.spatial_pos)  # (1, S, 1, D)
        x = self.linear(x) + self.pe_time[:, :, :x.size(2)] + embedded_space  # (N, S, T, D)
        return self.norm(x)


def TimeAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the time axis
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, T, D)
        mask: of size (T (query), T (key)) specifying locations (which key) the query can and cannot attend to
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, S, T, T)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        scores = scores.masked_fill(mask[None, None, None], float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)  # (N, h, S, T, D)


def SpaceAttention(query, key, value, mask=None, dropout=None):
    """
    attention over the two space axes
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, T, D)
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(2, 3)  # (N, h, T, S, D)
    key = key.transpose(2, 3)  # (N, h, T, S, D)
    value = value.transpose(2, 3)  # (N, h, T, S, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, T, S, S)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 3)  # (N, h, S, T_q, D)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (N, S, T, D)
        Returns:
            (N, S, T, D)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        # (N, h, S, T, d_k)
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        # (N, h, S, T, d_k)
        x = self.attn(query, key, value, mask=mask, dropout=self.dropout)

        # (N, S, T, D)
        x = x.permute(0, 2, 3, 1, 4).contiguous() \
             .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        return self.linears[-1](x)

