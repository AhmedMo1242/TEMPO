import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Local_Weighting(nn.Module):
    def __init__(self, input_size):
        super(Local_Weighting, self).__init__()
        self.conv = nn.Conv1d(
            input_size, input_size, kernel_size=5, stride=1, padding=2
        )
        self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
        # Small random initialization instead of all-zeros
        # to avoid NaN in InstanceNorm (0/0)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")
        with torch.no_grad():
            self.conv.weight.mul_(0.01)

    def forward(self, x):
        out = self.conv(x)
        return x + x * (torch.sigmoid(self.insnorm(out)) - 0.5)


class Temporal_LiftPool(nn.Module):
    def __init__(self, input_size, kernel_size=2):
        super(Temporal_LiftPool, self).__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.Conv1d(
                input_size,
                input_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_size,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        self.updater = nn.Sequential(
            nn.Conv1d(
                input_size,
                input_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_size,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        # Standard initialization (near zero but not exactly zero)
        for m in [self.predictor[2], self.updater[2]]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        self.weight1 = Local_Weighting(input_size)
        self.weight2 = Local_Weighting(input_size)

    def forward(self, x):
        B, C, T = x.size()
        # Ensure T is even if kernel_size=2
        if T % self.kernel_size != 0:
            x = F.pad(x, (0, self.kernel_size - (T % self.kernel_size)))
            T = x.size(2)

        Xe = x[:, :, : T : self.kernel_size]
        Xo = x[:, :, 1 : T : self.kernel_size]
        d = Xo - self.predictor(Xe)
        s = Xe + self.updater(d)

        # Concatenate or weight? USTM uses weighting
        return self.weight1(s) + self.weight2(d)


class MSTCN_Skeleton(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MSTCN_Skeleton, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.streams = nn.ModuleList()

        # Stream 0: Conv1x1 + MaxPool3x1
        self.streams.append(
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            )
        )

        # Stream 1: Conv1x1
        self.streams.append(
            nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=1), nn.ReLU(inplace=True)
            )
        )

        # Streams 2–5: Conv1x1 + Conv3x1 with dilation = 1, 2, 3, 4
        for d in [1, 2, 3, 4]:
            self.streams.append(
                nn.Sequential(
                    nn.Conv1d(input_size, hidden_size, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        hidden_size, hidden_size, kernel_size=3, dilation=d, padding=d
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        # After concat
        self.post_conv1 = nn.Sequential(
            nn.Conv1d(hidden_size * 6, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.tlp1 = Temporal_LiftPool(hidden_size, kernel_size=2)

        self.post_conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.tlp2 = Temporal_LiftPool(hidden_size, kernel_size=2)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        # MSTCN does 2x downsampling twice via LiftPool
        # Each LiftPool pads odd lengths, effectively doing ceil(T/2)
        feat_len = torch.div(feat_len + 1, 2, rounding_mode="floor")
        feat_len = torch.div(feat_len + 1, 2, rounding_mode="floor")
        feat_len = torch.clamp(feat_len, min=1)
        return feat_len

    def forward(self, x, lgt):
        # x: (B, C, T)
        stream_outputs = [stream(x) for stream in self.streams]
        x = torch.cat(stream_outputs, dim=1)  # (B, hidden_size * 6, T)

        x = self.post_conv1(x)
        x = self.tlp1(x)  # T / 2

        x = self.post_conv2(x)
        x = self.tlp2(x)  # T / 4

        lgt = self.update_lgt(lgt)
        # The expected output should be (T, B, C) and feature lengths
        # In MSLR, TemporalConv.forward returns permute(2, 0, 1) which is (T, B, C)
        return {
            "visual_feat": x.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
