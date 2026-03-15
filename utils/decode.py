"""CTC decoder: greedy (max) and prefix beam search.

Pure Python/PyTorch implementation — no external C++ libraries required.
Supports optional temperature scaling and test-time augmentation (TTA).
"""

import torch
import numpy as np
import torch.nn.functional as F
from itertools import groupby
from collections import defaultdict


class Decode:
    """CTC decoder with greedy and beam-search modes.

    Args:
        gloss_dict: Dictionary with ``id2gloss`` / ``gloss2id`` mappings.
        num_classes: Vocabulary size including the CTC blank.
        search_mode: ``"max"`` for greedy or ``"beam"`` for prefix beam search.
        blank_id: Index of the CTC blank token (default 0).
        beam_width: Number of active hypotheses in beam search.
        top_k: Per-timestep token pruning width.
        length_penalty: Additive length penalty (unused — kept for config compat).
        temperature: Logit temperature scaling before softmax.
        use_tta: Enable test-time augmentation (temporal flip + speed + shift).
    """

    def __init__(
        self,
        gloss_dict,
        num_classes,
        search_mode,
        blank_id=0,
        beam_width=30,
        top_k=40,
        length_penalty=0.0,
        temperature=1.0,
        use_tta=False,
    ):
        self.i2g_dict = {int(k): v["gloss"] for k, v in gloss_dict["id2gloss"].items()}
        self.g2i_dict = {k: int(v["index"]) for k, v in gloss_dict["gloss2id"].items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.top_k = top_k
        self.temperature = temperature
        self.use_tta = use_tta

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """Main entry point.

        Args:
            nn_output: (B, T, C) logits or probabilities.
            vid_lgt: Per-sample sequence lengths.
            batch_first: Whether the batch dimension is first.
            probs: If True, ``nn_output`` is already a probability distribution.
        """
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)

        if self.use_tta:
            return self._decode_with_tta(nn_output, vid_lgt, probs)

        if self.search_mode == "max":
            return self._greedy_decode(nn_output, vid_lgt)
        return self._beam_search(nn_output, vid_lgt, probs)

    # ------------------------------------------------------------------
    # Greedy (max) decoding
    # ------------------------------------------------------------------
    def _greedy_decode(self, nn_output, vid_lgt):
        if not isinstance(vid_lgt, torch.Tensor):
            vid_lgt = torch.tensor(vid_lgt)
        vid_lgt = vid_lgt.cpu().int()

        index_list = torch.argmax(nn_output, dim=-1).cpu().numpy()
        ret_list = []

        for i in range(nn_output.size(0)):
            collapsed = [x[0] for x in groupby(index_list[i][: vid_lgt[i]])]
            filtered = [x for x in collapsed if x != self.blank_id]
            ret_list.append(
                [
                    (self.i2g_dict.get(int(g), "UNK"), idx)
                    for idx, g in enumerate(filtered)
                ]
            )
        return ret_list

    # ------------------------------------------------------------------
    # Prefix beam search
    # ------------------------------------------------------------------
    def _beam_search(self, nn_output, vid_lgt, probs=False):
        if not isinstance(vid_lgt, torch.Tensor):
            vid_lgt = torch.tensor(vid_lgt)
        vid_lgt = vid_lgt.cpu().int()

        if not probs:
            nn_output = (
                nn_output / self.temperature if self.temperature != 1.0 else nn_output
            )
            nn_output = nn_output.softmax(-1)

        ret_list = []
        for i in range(nn_output.size(0)):
            seq_probs = nn_output[i, : vid_lgt[i]].cpu().numpy()

            # Each prefix maps to (prob_ending_blank, prob_ending_non_blank).
            beams = {(): (1.0, 0.0)}

            for t in range(len(seq_probs)):
                new_beams = defaultdict(lambda: (0.0, 0.0))
                top_indices = np.argsort(seq_probs[t])[-self.top_k :]

                for c in top_indices:
                    p_c = seq_probs[t, c]
                    for prefix, (p_b, p_nb) in beams.items():
                        if c == self.blank_id:
                            old = new_beams[prefix]
                            new_beams[prefix] = (old[0] + (p_b + p_nb) * p_c, old[1])
                        else:
                            last = prefix[-1] if prefix else None
                            if c == last:
                                # Same token: extend only from blank.
                                old = new_beams[prefix]
                                new_beams[prefix] = (old[0], old[1] + p_nb * p_c)
                                ext = prefix + (c,)
                                old2 = new_beams[ext]
                                new_beams[ext] = (old2[0], old2[1] + p_b * p_c)
                            else:
                                ext = prefix + (c,)
                                old = new_beams[ext]
                                new_beams[ext] = (old[0], old[1] + (p_b + p_nb) * p_c)

                beams = dict(
                    sorted(new_beams.items(), key=lambda x: sum(x[1]), reverse=True)[
                        : self.beam_width
                    ]
                )

            best = max(beams.items(), key=lambda x: sum(x[1]))[0]
            ret_list.append(
                [(self.i2g_dict.get(int(g), "UNK"), idx) for idx, g in enumerate(best)]
            )
        return ret_list

    # ------------------------------------------------------------------
    # Test-time augmentation
    # ------------------------------------------------------------------
    def _decode_with_tta(self, nn_output, vid_lgt, probs=False):
        """Average probabilities over augmented views then beam-search."""
        B, T, C = nn_output.shape

        if not probs:
            logits = (
                nn_output / self.temperature if self.temperature != 1.0 else nn_output
            )
            base_probs = logits.softmax(-1)
        else:
            base_probs = nn_output

        views = [base_probs]

        # Temporal flip.
        views.append(torch.flip(base_probs, dims=[1]))

        # Speed perturbation (0.9x / 1.1x resampled back to T).
        bct = base_probs.permute(0, 2, 1)  # (B, C, T)
        for factor in (0.9, 1.1):
            pert = F.interpolate(
                bct, scale_factor=factor, mode="linear", align_corners=False
            )
            pert = F.interpolate(pert, size=T, mode="linear", align_corners=False)
            views.append(pert.permute(0, 2, 1))

        # Small temporal shifts (±1 frame).
        views.append(torch.roll(base_probs, shifts=1, dims=1))
        views.append(torch.roll(base_probs, shifts=-1, dims=1))

        avg_probs = torch.stack(views, dim=0).mean(dim=0)
        return self._beam_search(avg_probs, vid_lgt, probs=True)


# Backward-compatible alias used by slr_network.py
EnhancedDecode = Decode
