"""
Pytorch Implementation of Conv-TT-LSTM (Su et al., 2020) (https://arxiv.org/abs/2002.09131)
The code is mainly based on https://github.com/NVlabs/conv-tt-lstm, with minor changes.
"""

import torch
import torch.nn as nn


class ConvTTLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, order, steps, ranks,
                 kernel_size, bias):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.steps = steps
        self.order = order
        self.lags = steps - order + 1

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, bias=bias, padding=padding)

        Conv3d = lambda in_channels, out_channels: nn.Conv3d(in_channels=in_channels, out_channels=out_channels, bias=bias,
                      kernel_size=kernel_size + (self.lags, ), padding=(padding[0], padding[1], 0))

        self.layers = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels =ranks if l < order - 1 else ranks + input_channels, 
                out_channels=ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv3d(in_channels=hidden_channels, out_channels=ranks))

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0

        self.cell_states = torch.zeros(batch_size,
            self.hidden_channels, height, width, device=device)

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.stack(input_states, dim=-1)
            input_states = self.layers_[l](input_states)
            input_states = torch.squeeze(input_states, dim=-1)

            if l == 0:
                temp_states = input_states
            else:
                temp_states = input_states + self.layers[l-1](temp_states)

        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states = f * self.cell_states + i * g
        outputs = o * torch.tanh(self.cell_states)
        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps

        return outputs


class ConvTTLSTMNet(nn.Module):
    def __init__(self, order, steps, ranks, kernel_size, bias,
                 hidden_channels, layers_per_block, skip_stride,
                 input_dim, output_dim):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.num_blocks = len(layers_per_block)
        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride
        self.output_dim = output_dim

        Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
            input_channels=in_channels, hidden_channels=out_channels,
            order=order, steps=steps, ranks=ranks, 
            kernel_size=kernel_size, bias=bias)

        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(self.layers_per_block[b]):
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0:
                    channels = input_dim
                else:
                    channels = hidden_channels[b-1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b-1-self.skip_stride] 

                lid = "b{}l{}".format(b, l)
                self.layers[lid] = Cell(channels, hidden_channels[b])

        channels = hidden_channels[-1]
        if self.num_blocks > self.skip_stride:
            channels += hidden_channels[-1-self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=12, future_frames=26, output_frames=37,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        """
        The Conv-TT-LSTM module, employed with scheduled sampling
        for multi-step spatio-temporalforecasting.
        The network is designed to predict the next frame based on the context in the current time step,
        and multi-step forecasts are made by recursively invoking the SAConvLSTMCell.
        The sst in the input time period are also used as the ground truth for training
        Args:
            input_x: input with size (N, T, H, W)
            input_frames: the number of input time steps
            future_frames: the number of target time steps for SST
            output_frames: the number of model output time steps, typically equal to
                           input_frames + future_frames - 1 (training) or future_frames (testing)
            teacher_forcing: specify if teacher forcing is used. Expect True (training), False (testing)
            scheduled_sampling_ratio: The sampling ratio used during scheduled sampling 
            train: specify whether or not the model is in the train mode
        Returns:
            outputs: the predicted SST with size (N, output_frames, H, W) for backward propagation
            nino_pred: the predicted nino with size (N, future_frames)
        """
        assert len(input_x.shape) == 4
        input_x = input_x[:, :, None]
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
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

            queue = []
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)
                    input_ = self.layers[lid](input_, first_step=first_step)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)

            if train or (t >= (input_frames - 1)):
                outputs[t] = self.layers["output"](input_)
            else:
                pass
        outputs = [x for x in outputs if x is not None]
        if train:
            assert len(outputs) == output_frames
        else:
            assert len(outputs) == future_frames

        outputs = torch.stack(outputs, dim=1)[:, :, 0]  # (N, 37, H, W)
        nino_pred = outputs[:, -future_frames:, 10:13, 19:30].mean(dim=[2, 3])  # (N, 26)
        nino_pred = nino_pred.unfold(dimension=1, size=3, step=1).mean(dim=2)  # (N, 24)
        return outputs, nino_pred
