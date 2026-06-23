"""Cross-Stream Consistency (CSC) loss.

Regularises the two-stream backbone by encouraging the static and
motion stream representations to agree via an InfoNCE-style contrastive
objective.  This prevents each stream from specialising on
signer-specific patterns that do not transfer to unseen signers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStreamConsistencyLoss(nn.Module):
    """InfoNCE contrastive loss between static and motion stream features.

    Each stream's temporal features are mean-pooled to a sample-level
    descriptor, L2-normalised, and matched across the batch with a
    temperature-scaled cross-entropy objective.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_static, feat_motion):
        """Compute CSC loss.

        Args:
            feat_static:  (B, T, C) — static stream temporal features.
            feat_motion:  (B, T, C) — motion stream temporal features.

        Returns:
            Scalar contrastive loss.
        """
        s = F.normalize(feat_static.mean(dim=1), dim=-1)
        m = F.normalize(feat_motion.mean(dim=1), dim=-1)

        logits = torch.matmul(s, m.T) / self.temperature
        labels = torch.arange(s.size(0), device=s.device)

        return F.cross_entropy(logits, labels)
