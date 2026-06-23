"""
CSV-based Word Error Rate (WER) evaluation.

Computes standard WER (ins=1, del=1, sub=1) directly from CSV files.
No CTM/STM files needed — just CSV with columns: id, gloss.

Usage as module:
    from evaluation.wer import compute_wer, evaluate_csv

Usage from CLI:
    python -m evaluation.wer --ref ref.csv --hyp pred.csv
"""

import numpy as np


def edit_distance(ref, hyp):
    """Standard edit distance with (ins=1, del=1, sub=1). Returns (cost, n_sub, n_del, n_ins)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])

    # back-trace for operation counts
    i, j = n, m
    n_sub = n_del = n_ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            n_sub += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            n_del += 1
            i -= 1
        else:
            n_ins += 1
            j -= 1

    return int(d[n, m]), n_sub, n_del, n_ins


def compute_wer(ref_dict, hyp_dict):
    """Compute corpus WER from dicts {id: "gloss words"}.

    Returns:
        wer_percent (float): WER as percentage
        details (dict): {total_ref, total_errors, n_sub, n_del, n_ins, n_correct, n_samples}
    """
    total_ref = 0
    total_errors = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    n_correct = 0
    n_samples = 0

    for vid_id, ref_gloss in ref_dict.items():
        ref_tokens = str(ref_gloss).strip().split()
        hyp_gloss = hyp_dict.get(vid_id, "")
        hyp_tokens = str(hyp_gloss).strip().split() if hyp_gloss else []

        cost, n_sub, n_del, n_ins = edit_distance(ref_tokens, hyp_tokens)
        total_ref += len(ref_tokens)
        total_errors += cost
        total_sub += n_sub
        total_del += n_del
        total_ins += n_ins
        if cost == 0:
            n_correct += 1
        n_samples += 1

    wer_pct = (total_errors / total_ref * 100) if total_ref > 0 else 0.0

    return wer_pct, {
        "total_ref": total_ref,
        "total_errors": total_errors,
        "n_sub": total_sub,
        "n_del": total_del,
        "n_ins": total_ins,
        "n_correct": n_correct,
        "n_samples": n_samples,
    }


def load_csv_as_dict(csv_path, sep=","):
    """Load CSV with id|gloss or id,gloss into {id: gloss} dict."""
    import csv as csv_module

    result = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        # Auto-detect separator
        first_line = f.readline().strip()
        if "|" in first_line:
            sep = "|"
        f.seek(0)

        reader = csv_module.DictReader(f, delimiter=sep)
        for row in reader:
            vid_id = row.get("id", "").strip()
            gloss = row.get("gloss", "").strip()
            if vid_id:
                result[vid_id] = gloss
    return result


def evaluate_csv(ref_csv, hyp_csv):
    """Evaluate WER from two CSV files. Returns wer_percent."""
    ref_dict = load_csv_as_dict(ref_csv)
    hyp_dict = load_csv_as_dict(hyp_csv)
    wer_pct, details = compute_wer(ref_dict, hyp_dict)
    return wer_pct, details


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSV-based WER evaluation")
    parser.add_argument(
        "--ref", required=True, help="Reference CSV (id|gloss or id,gloss)"
    )
    parser.add_argument("--hyp", required=True, help="Hypothesis CSV (id,gloss)")
    args = parser.parse_args()

    wer_pct, details = evaluate_csv(args.ref, args.hyp)
    print(f"WER: {wer_pct:.2f}%")
    print(f"  Samples: {details['n_samples']}")
    print(f"  Ref tokens: {details['total_ref']}")
    print(
        f"  Errors: {details['total_errors']} (Sub={details['n_sub']}, Del={details['n_del']}, Ins={details['n_ins']})"
    )
    print(f"  Correct sentences: {details['n_correct']}/{details['n_samples']}")
