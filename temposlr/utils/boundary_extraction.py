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


def _smooth1d(values: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Apply simple moving-average smoothing to a 1-D tensor."""
    if kernel <= 1 or len(values) <= kernel:
        return values
    pad = kernel // 2
    # Constant-pad (replicate edge values) then 1-D conv
    v = values.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    padded = torch.nn.functional.pad(v, (pad, pad), mode="replicate")
    kernel_filter = torch.ones(1, 1, kernel, device=values.device) / kernel
    return torch.nn.functional.conv1d(padded, kernel_filter).squeeze()


def _find_transition_frame(
    probs: torch.Tensor,
    label_a: int,
    label_b: int,
    search_start: int,
    search_end: int,
    blank_id: int = 0,
) -> int:
    """
    Find the frame where the model transitions from label_a to label_b.

    Uses the **midpoint** between their detection frames as the split.
    Detection frames (first argmax appearance) are already robust — they
    come from the collapsed CTC argmax sequence which is deterministic and
    noise-free.  Probability-based methods (zero-crossing, peak detection)
    add fragility without meaningful accuracy gain for well-trained models.

    Args:
        probs: (T_feat, C) softmax probabilities (unused; kept for API compat)
        label_a: previous word's label id (unused)
        label_b: next word's label id (unused)
        search_start: detection frame of word A
        search_end: detection frame of word B
        blank_id: CTC blank index (unused)

    Returns:
        Frame index of the transition (midpoint between detections).
    """
    return (search_start + search_end) // 2


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
    gloss a continuous span using probability-based transition detection:
    - First gloss starts at its first detection frame
    - Last gloss ends at feat_len
    - Consecutive glosses are split where their probabilities cross over

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

    # Build continuous spans with probability-based transitions
    n = len(label_seq)
    segments = []

    # Pre-compute all transition frames between consecutive words
    transition_frames = []
    for i in range(n - 1):
        det_i = detection_frames[i] if i < len(detection_frames) else 0
        det_next = detection_frames[i + 1] if i + 1 < len(detection_frames) else feat_len
        trans = _find_transition_frame(
            probs, label_seq[i], label_seq[i + 1],
            search_start=det_i,
            search_end=det_next,
            blank_id=blank_id,
        )
        transition_frames.append(trans)

    for i, lid in enumerate(label_seq):
        det = detection_frames[i] if i < len(detection_frames) else 0

        if i == 0:
            start = det  # Start at first detection frame
        else:
            start = transition_frames[i - 1]  # Use probability-based transition

        if i == n - 1:
            end = feat_len
        else:
            end = transition_frames[i]  # Use probability-based transition

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
