import copy
import re
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        # support both list and string conv_type; allow tokens like 'P' or 'K3'
        if isinstance(conv_type, str):
            self.kernel_size = conv_type.split("-")
        else:
            self.kernel_size = list(conv_type)

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            m = re.match(r"([PK])(\d*)", ks)
            if not m:
                continue
            t, num = m.group(1), m.group(2)
            # defaults: pooling -> 2, kernel -> 3
            k = int(num) if num != "" else (2 if t == "P" else 3)
            if t == "P":
                modules.append(nn.MaxPool1d(kernel_size=k, ceil_mode=False))
            elif t == "K":
                modules.append(
                    nn.Conv1d(
                        input_sz,
                        self.hidden_size,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                    )
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            m = re.match(r"([PK])(\d*)", ks)
            if not m:
                continue
            t, num = m.group(1), m.group(2)
            k = int(num) if num != "" else (2 if t == "P" else 3)
            if t == "P":
                feat_len = torch.div(feat_len, k).long()
        feat_len = torch.clamp(feat_len, min=1)
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        if visual_feat.shape[2] == 0:
            visual_feat = torch.zeros(
                visual_feat.shape[0], visual_feat.shape[1], 1, device=visual_feat.device
            )
        lgt = torch.clamp(lgt, min=1)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
