"""
Post-processing v5 for CSLR predictions.

Combined score: normalized DL + Jaccard distance penalty.

  score(pred, cand) = alpha * norm_DL(pred, cand)
                    + (1 - alpha) * (1 - jaccard(pred, cand))

  jaccard(A, B) = |set(A) ∩ set(B)| / |set(A) ∪ set(B)|

Why this helps on top of v4:
  - Normalized DL (even avg-len) can still prefer candidates that share
    zero words with the prediction if they happen to be a similar length.
  - Jaccard adds a BAG-OF-WORDS overlap constraint: candidates that share
    more sign tokens with the prediction are explicitly rewarded.
  - The two terms are complementary:
      DL  → captures sequence order, insertions, deletions, transpositions
      Jaccard → captures unordered word inventory overlap

  alpha=0.6 means DL drives 60% of the score, Jaccard contributes 40%.
  (Tune with --alpha flag)

Usage:
    python dp_postprocess_v5.py [--input 1/test.csv] [--output dp5/test.csv]
    python dp_postprocess_v5.py --alpha 0.7   # trust DL more
    python dp_postprocess_v5.py --alpha 0.5   # equal weight
"""

import argparse
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Damerau-Levenshtein (identical to v2/v4)
# ---------------------------------------------------------------------------


def damerau_levenshtein(seq_a: List[str], seq_b: List[str]) -> int:
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    INF = len_a + len_b + 1
    d = [[INF] * (len_b + 2) for _ in range(len_a + 2)]
    d[0][0] = INF
    for i in range(len_a + 1):
        d[i + 1][0] = INF
        d[i + 1][1] = i
    for j in range(len_b + 1):
        d[0][j + 1] = INF
        d[1][j + 1] = j

    last_row: dict = {}
    for i in range(1, len_a + 1):
        tok_a = seq_a[i - 1]
        last_col = 0
        for j in range(1, len_b + 1):
            tok_b = seq_b[j - 1]
            i2 = last_row.get(tok_b, 0)
            j2 = last_col
            if tok_a == tok_b:
                cost = 0
                last_col = j
            else:
                cost = 1
            d[i + 1][j + 1] = min(
                d[i][j] + cost,
                d[i + 1][j] + 1,
                d[i][j + 1] + 1,
                d[i2][j2] + (i - i2 - 1) + 1 + (j - j2 - 1),
            )
        last_row[tok_a] = i
    return d[len_a + 1][len_b + 1]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def best_match_worker(
    args: Tuple[List[str], List[List[str]], List[set], float],
) -> Tuple[str, int]:
    pred_tokens, vocab_tokens, vocab_sets, alpha = args
    best_score = float("inf")
    best_raw = float("inf")
    best_sent = pred_tokens
    pred_len = len(pred_tokens)
    pred_set = set(pred_tokens)

    for cand_tokens, cand_set in zip(vocab_tokens, vocab_sets):
        cand_len = len(cand_tokens)
        max_len = max(pred_len, cand_len) if max(pred_len, cand_len) > 0 else 1

        # Fast Jaccard lower-bound skip: if jaccard component alone can't beat best_score
        jac = jaccard(pred_set, cand_set)
        jac_component = (1.0 - alpha) * (1.0 - jac)  # this part of score is fixed

        # Lower bound for DL component: abs(len_diff) / max_len
        dl_lb_norm = abs(pred_len - cand_len) / max_len
        lb = alpha * dl_lb_norm + jac_component
        if lb >= best_score:
            continue

        raw = damerau_levenshtein(pred_tokens, cand_tokens)
        norm_dl = raw / max_len
        score = alpha * norm_dl + jac_component

        if score < best_score:
            best_score = score
            best_raw = raw
            best_sent = cand_tokens

        if score == 0.0:
            break

    return " ".join(best_sent), int(best_raw) if best_raw != float("inf") else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DP post-processing v5 (DL + Jaccard combined score)"
    )
    parser.add_argument("--input", default="1/test.csv", help="Prediction CSV")
    parser.add_argument("--output", default="dp5/test.csv", help="Output CSV")
    parser.add_argument("--train", default="train.csv", help="Train CSV (id|gloss)")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Weight for DL term (1-alpha goes to Jaccard). Default: 0.6",
    )
    args = parser.parse_args()

    print(f"Loading train vocabulary from: {args.train}")
    train_df = pd.read_csv(args.train, sep="|")
    unique_sentences = train_df["gloss"].dropna().unique().tolist()
    vocab_tokens: List[List[str]] = [str(s).split() for s in unique_sentences]
    vocab_sets: List[set] = [set(t) for t in vocab_tokens]
    print(f"  → {len(vocab_tokens)} unique sentences in vocabulary")

    print(f"\nLoading predictions from: {args.input}")
    pred_df = pd.read_csv(args.input)
    print(f"  → {len(pred_df)} predictions")
    print(
        f"  → alpha (DL weight) = {args.alpha}  |  Jaccard weight = {1-args.alpha:.1f}"
    )

    pred_tokens_list: List[List[str]] = []
    for gloss in pred_df["gloss"]:
        if pd.isna(gloss) or str(gloss).strip() == "":
            pred_tokens_list.append([])
        else:
            pred_tokens_list.append(str(gloss).split())

    print(
        f"\nMatching {len(pred_tokens_list)} predictions → {len(vocab_tokens)} candidates "
        f"using {args.workers} workers …"
    )
    work_items = [(p, vocab_tokens, vocab_sets, args.alpha) for p in pred_tokens_list]

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(best_match_worker, work_items, chunksize=50),
                total=len(work_items),
                desc="DL+Jaccard matching",
                unit="pred",
            )
        )

    corrected, total_changed, total_cost, samples = [], 0, 0, []
    for i, (matched_gloss, dist) in enumerate(results):
        original = " ".join(pred_tokens_list[i])
        if matched_gloss != original:
            total_changed += 1
            if len(samples) < 10:
                samples.append((dist, original, matched_gloss))
        total_cost += dist
        corrected.append(matched_gloss)

    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    out_df = pred_df.copy()
    out_df["gloss"] = corrected
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*55}")
    print(f"  Algorithm           : DL (max-len norm) + Jaccard  [alpha={args.alpha}]")
    print(f"  Predictions changed : {total_changed} / {len(pred_df)}")
    print(f"  Total edit distance : {total_cost}")
    print(f"  Avg edit distance   : {total_cost / len(pred_df):.4f}")
    print(f"  Output saved to     : {args.output}")
    print(f"{'='*55}")
    print("\nSample corrections (original → matched):")
    for dist, orig, matched in samples:
        print(f"  [{dist}] {orig} → {matched}")


if __name__ == "__main__":
    main()
