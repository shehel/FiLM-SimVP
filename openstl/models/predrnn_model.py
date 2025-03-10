import torch
import torch.nn as nn

from openstl.modules import SpatioTemporalLSTMCell
import pdb

class PredRNN_Model(nn.Module):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNN_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width,
                                       configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_lo1 = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_lo2 = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_hi1 = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_hi2 = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_m = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    try:
                        net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                              (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen
                    except:
                        pdb.set_trace()
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_lo1 = self.conv_lo1(h_t[self.num_layers - 1])
            x_lo2 = self.conv_lo2(h_t[self.num_layers - 1])
            x_hi1 = self.conv_hi1(h_t[self.num_layers - 1])
            x_hi2 = self.conv_hi2(h_t[self.num_layers - 1])
            x_gen = self.conv_m(h_t[self.num_layers - 1])
            x = torch.cat((x_lo1.unsqueeze(1), x_lo2.unsqueeze(1), x_gen.unsqueeze(1), x_hi1.unsqueeze(1), x_hi2.unsqueeze(1)), dim=1)
            # add an empty dimension at first axis
            #x = x.reshape(batch, 3, self.out_ts, self.out_ch, H, W)
            next_frames.append(x)

        # [length, batch, quantiles, channel, height, width] -> [batch,quantiles, length, height, width, channel]

        next_frames = torch.stack(next_frames, dim=0).permute(1, 2, 0, 4, 5, 3).contiguous()

        # if kwargs.get('return_loss', True):
        #     loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        # else:
        loss = None

        return next_frames, loss
