"""Stochastic Sequence Depth (SSD).

Randomly bypasses a temporal module during training to prevent
co-adaptation between the backbone and the temporal convolution.
At test time the full network is always used.
"""

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Drop an entire module's output with probability ``prob`` during training.

    When dropped, outputs are zeroed while shapes (including ``feat_len``)
    are preserved so that downstream layers receive consistent tensor sizes.
    """

    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(self, x, fn):
        """Apply *fn(x)* and optionally zero the result.

        Args:
            x: Input tensor.
            fn: Callable that produces the module output (may return a dict
                with ``visual_feat`` and ``feat_len`` keys).
        """
        out = fn(x)

        if not self.training:
            return out

        # Keep shapes consistent across DataParallel replicas.
        if torch.cuda.device_count() > 1:
            return out

        if torch.rand(1).item() > self.prob:
            return out

        # Zero features but keep metadata such as feat_len.
        if isinstance(out, dict):
            return {
                "visual_feat": torch.zeros_like(out["visual_feat"]),
                "feat_len": out["feat_len"],
            }
        return torch.zeros_like(out)
