import torch
import torch.nn as nn
import torch.nn.functional as F


class TAPEAdapter(nn.Module):
    """
    TAPE: Temporal Adapter with Positional Embeddings
    Encodes temporal dynamics into spatial-focused features for Skeleton-based SLR.
    Adapted from USTM (LVM-STeM) 2025.

    This module uses a bottleneck structure (Linear down -> Depthwise Conv -> Linear up)
    to efficiently model temporal dependencies with minimal parameter overhead.
    """

    def __init__(
        self, in_channels, adapter_channels=64, kernel_size=3, max_frames=1024
    ):
        super().__init__()
        self.in_channels = in_channels
        self.adapter_channels = adapter_channels

        self.norm1 = nn.LayerNorm(in_channels)
        self.down_proj = nn.Linear(in_channels, adapter_channels)

        # Temporal Positional Encoding
        self.temporal_pos_emb = nn.Parameter(torch.zeros(max_frames, adapter_channels))

        # Temporal Depthwise Convolution
        self.dw_conv = nn.Conv1d(
            adapter_channels,
            adapter_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=adapter_channels,
        )

        self.norm2 = nn.LayerNorm(adapter_channels)
        self.up_proj = nn.Linear(adapter_channels, in_channels)

        # Initialize weights
        self._init_weights()

        print(f"USING TAPEAdapter (in: {in_channels}, adapter: {adapter_channels})")

    def _init_weights(self):
        # Use standard initialization with very small gain for stability
        # Avoid all-zero initialization which causes LayerNorm to produce NaN
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)

        # Depthwise conv: initialization near identity or small random
        nn.init.kaiming_normal_(
            self.dw_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        with torch.no_grad():
            self.dw_conv.weight.mul_(0.01)  # Scale down to start near identity

        if self.dw_conv.bias is not None:
            nn.init.constant_(self.dw_conv.bias, 0.0)
        nn.init.constant_(self.down_proj.bias, 0.0)
        nn.init.constant_(self.up_proj.bias, 0.0)
        # Initialize positional embeddings with truncated normal
        nn.init.trunc_normal_(self.temporal_pos_emb, std=0.02)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        identity = x

        x = self.norm1(x)
        x = self.down_proj(x)

        # Add Temporal Positional Encoding
        if T <= self.temporal_pos_emb.size(0):
            pos_emb = self.temporal_pos_emb[:T]
        else:
            # Interpolate if sequence is longer than max_frames
            pos_emb = (
                F.interpolate(
                    self.temporal_pos_emb.unsqueeze(0).transpose(1, 2),
                    size=T,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .squeeze(0)
            )

        x = x + pos_emb.unsqueeze(0)

        # Temporal Depthwise Conv
        x = x.transpose(1, 2)  # (B, C_adapter, T)
        x = self.dw_conv(x)
        x = x.transpose(1, 2)  # (B, T, C_adapter)

        x = self.norm2(x)
        x = self.up_proj(x)

        return identity + x
