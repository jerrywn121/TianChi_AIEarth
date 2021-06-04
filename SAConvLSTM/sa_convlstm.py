"""
Author: written by Jiacheng WU
The model architecture is adopted from SA-ConvLSTM (Lin et al., 2020) 
(https://ojs.aaai.org/index.php/AAAI/article/view/6819)
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def attn(query, key, value):
    """
    Apply attention over the spatial dimension (S)
    Args:
        query, key, value: (N, C, S)
    Returns:
        output of the same size
    """
    scores = query.transpose(1, 2) @ key / math.sqrt(query.size(1))  # (N, S, S)
    attn = F.softmax(scores, dim=-1)
    output = attn @ value.transpose(1, 2)
    return output.transpose(1, 2)  # (N, C, S)


class SAAttnMem(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size):
        """
        The self-attention memory module added to ConvLSTM
        """
        super().__init__()
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.d_model = d_model
        self.input_dim = input_dim
        self.conv_h = nn.Conv2d(input_dim, d_model*3, kernel_size=1)
        self.conv_m = nn.Conv2d(input_dim, d_model*2, kernel_size=1)
        self.conv_z = nn.Conv2d(d_model*2, d_model, kernel_size=1)
        self.conv_output = nn.Conv2d(input_dim+d_model, input_dim*3, kernel_size=kernel_size, padding=pad)

    def forward(self, h, m):
        hq, hk, hv = torch.split(self.conv_h(h), self.d_model, dim=1)
        mk, mv = torch.split(self.conv_m(m), self.d_model, dim=1)
        N, C, H, W = hq.size()
        Zh = attn(hq.view(N, C, -1), hk.view(N, C, -1), hv.view(N, C, -1))  # (N, S, C)
        Zm = attn(hq.view(N, C, -1), mk.view(N, C, -1), mv.view(N, C, -1))  # (N, S, C)
        Z = self.conv_z(torch.cat([Zh.view(N, C, H, W), Zm.view(N, C, H, W)], dim=1))
        i, g, o = torch.split(self.conv_output(torch.cat([Z, h], dim=1)), self.input_dim, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        m_next = i * g + (1 - i) * m
        h_next = torch.sigmoid(o) * m_next
        return h_next, m_next


class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_attn, kernel_size):
        """
        The SA-ConvLSTM cell module. Same as the ConvLSTM cell except with the
        self-attention memory module and the M added
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=pad)
        self.sa = SAAttnMem(input_dim=hidden_dim, d_model=d_attn, kernel_size=kernel_size)

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.memory_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        combined = torch.cat([inputs, self.hidden_state], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # novel for sa-convlstm
        self.hidden_state, self.memory_state = self.sa(self.hidden_state, self.memory_state)
        return self.hidden_state


class SAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_attn, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The self-attention ConvLSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, C, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if the teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling 
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        assert len(input_x.shape) == 5
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)
                input_ = input_x[:, t].to(device) * mask + outputs[t-1] * (1 - mask)
            first_step = (t == 0)
            input_ = input_.float()

            first_step = (t == 0)
            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.conv_output(input_)

        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]  # (N, 37, H, W)
        nino_pred = outputs[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
