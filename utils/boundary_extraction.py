"""
Gloss boundary extraction from CTC logits.

Given per-frame logits from the model, extracts continuous start/end frame
for each predicted gloss using argmax decoding + midpoint interpolation.

The model's TemporalConv downsamples time by 4x, so each feature frame
maps to ~4 original input frames.  ``feat_to_orig()`` handles the mapping.

Usage::

    from utils.boundary_extraction import extract_boundaries, feat_to_orig

    segments = extract_boundaries(logits, feat_len, norm_scale=32)
    # segments: [(label_id, start_feat, end_feat), ...]

    for lid, sf, ef in segments:
        so, eo = feat_to_orig(sf, ef, T_orig)
        print(f"gloss_id={lid}  frames [{so}, {eo}]")
"""
import torch
from typing import List, Dict, Tuple


def _decode_sequence(
    logits: torch.Tensor,
    feat_len: int,
    blank_id: int = 0,
    norm_scale: float = 1.0,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """
    Decode CTC output to get label sequence and per-frame argmax.

    Args:
        logits: (T_feat, C) raw cosine-similarity logits
        feat_len: number of valid feature frames
        blank_id: CTC blank index (default 0)
        norm_scale: multiplier applied to logits before softmax

    Returns:
        label_seq: collapsed non-blank label ids in order
        probs: (T_feat, C) softmax probabilities
        token_ids: list of per-frame argmax token ids
    """
    valid = logits[:feat_len] * norm_scale
    probs = torch.softmax(valid, dim=-1)
    token_ids = torch.argmax(probs, dim=-1).tolist()

    # Collapse consecutive duplicates
    collapsed = []
    prev = None
    for tok in token_ids:
        if tok != prev:
            collapsed.append(tok)
            prev = tok
    label_seq = [tok for tok in collapsed if tok != blank_id]
    return label_seq, probs, token_ids


def _find_detection_frames(
    token_ids: List[int],
    label_seq: List[int],
    blank_id: int = 0,
) -> List[int]:
    """
    Find the first frame where each label in *label_seq* appears.

    Args:
        token_ids: per-frame argmax token ids
        label_seq: decoded label sequence (no blanks)
        blank_id: CTC blank index

    Returns:
        List of frame indices (one per label in *label_seq*).
    """
    frames = []
    expected = 0
    prev = None
    for t, tok in enumerate(token_ids):
        if tok != prev:
            prev = tok
            if tok != blank_id and expected < len(label_seq) and tok == label_seq[expected]:
                frames.append(t)
                expected += 1
                if expected >= len(label_seq):
                    break
    return frames


def extract_boundaries(
    logits: torch.Tensor,
    feat_len: int,
    gloss_dict=None,
    blank_id: int = 0,
    norm_scale: float = 1.0,
) -> List[Tuple[int, int, int]]:
    """
    Extract continuous gloss boundaries from CTC logits.

    Uses argmax decoding to find the label sequence, then assigns each
    gloss a continuous span via midpoint interpolation between detections:
    - First gloss starts at feature frame 0
    - Last gloss ends at feat_len
    - Consecutive glosses are split at the midpoint of their detection frames

    Args:
        logits: (T_feat, C) raw cosine-similarity logits from the model
        feat_len: number of valid feature frames
        gloss_dict: unused (kept for API compat); pass for validation
        blank_id: CTC blank index (default 0)
        norm_scale: logit multiplier before softmax (training default: 32)

    Returns:
        List of (label_id, start_feat, end_feat_exclusive) tuples.
        Empty list if no glosses detected.
    """
    label_seq, probs, token_ids = _decode_sequence(logits, feat_len, blank_id, norm_scale)
    if not label_seq:
        return []

    detection_frames = _find_detection_frames(token_ids, label_seq, blank_id)

    # Build continuous spans
    n = len(label_seq)
    segments = []
    for i, lid in enumerate(label_seq):
        det = detection_frames[i] if i < len(detection_frames) else 0

        if i == 0:
            start = 0
        else:
            prev_det = detection_frames[i - 1] if i - 1 < len(detection_frames) else 0
            start = (prev_det + det) // 2

        if i == n - 1:
            end = feat_len
        else:
            next_det = detection_frames[i + 1] if i + 1 < len(detection_frames) else feat_len
            end = (det + next_det) // 2

        if end <= start:
            end = start + 1

        segments.append((lid, start, end))
    return segments


# Backward-compat aliases
viterbi_boundaries = extract_boundaries
greedy_boundaries = extract_boundaries


def feat_to_orig(
    start_feat: int,
    end_feat: int,
    t_orig: int,
    left_pad: int = 6,
) -> Tuple[int, int]:
    """
    Map a feature-frame range back to original video frame range.

    The pipeline applies: pad left=6, then TemporalConv K3-P2-K3-P2 (4x down).
    So feature frame *f* corresponds to padded frames [4f, 4f+4), and padded
    frame *p* corresponds to original frame (p - left_pad).

    Args:
        start_feat: inclusive start feature frame
        end_feat: exclusive end feature frame
        t_orig: total frames in the original (pre-padding) video
        left_pad: number of left-padding frames (default 6)

    Returns:
        (start_orig, end_orig) inclusive frame indices in the original video.
    """
    start_orig = max(0, min(t_orig - 1, start_feat * 4 - left_pad))
    end_orig = max(start_orig, min(t_orig - 1, end_feat * 4 - left_pad - 1))
    return start_orig, end_orig
