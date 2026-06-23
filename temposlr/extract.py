#!/usr/bin/env python3
"""
Gloss boundary extraction from TEMPO skeleton inference.

Supports train/dev/test splits.

CLI:
    python -m temposlr.extract 02_0001                          # one sample (dev)
    python -m temposlr.extract 02_0001 02_0002 02_0003          # multiple
    python -m temposlr.extract --all --limit 10                  # first 10
    python -m temposlr.extract --split train --all --limit 10    # train split
    python -m temposlr.extract --split test --all --limit 10     # test split
    python -m temposlr.extract 02_0001 --json                    # JSON output
    python -m temposlr.extract 02_0001 --csv                     # CSV output

Library:
    from temposlr.extract import BoundaryExtractor
    ext = BoundaryExtractor(split="dev")
    segments = ext("02_0001")
"""
import os, sys, json, yaml, argparse, csv
import numpy as np
import torch

# Resolve paths relative to package root (TEMPO/)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BoundaryExtractor:
    """Extract gloss boundaries from TEMPO model inference.

    Supports train/dev/test splits via the `split` parameter.

    Usage:
        ext = BoundaryExtractor(config="configs/test.yaml", split="dev")
        segments = ext("02_0001")
    """

    # Split → video ID prefix mapping
    SPLIT_PREFIXES = {
        "train": "00",
        "dev": "02",
        "test": "08",
    }

    def __init__(
        self,
        config: str = "configs/test.yaml",
        gloss_dict_path: str = None,
        device: str = None,
        split: str = None,
    ):
        """Load model and build dataset feeder.

        Args:
            config: path to config YAML
            gloss_dict_path: path to gloss dict JSON (default: auto-detect)
            device: "cuda" or "cpu" (auto-detected if None)
            split: "train", "dev", or "test" (overrides config if set)
        """
        from temposlr.datasets.skeleton_feeder import SkeletonFeeder
        from temposlr import model as slr_network

        if gloss_dict_path is None:
            gloss_dict_path = os.path.join(
                _PKG_ROOT, "temposlr", "datasets", "mslr2026", "si_gloss_dict.json"
            )

        with open(config) as f:
            self.cfg = yaml.safe_load(f)
        with open(gloss_dict_path) as f:
            self.gloss_dict = json.load(f)

        # Determine split
        self.split = split or self.cfg.get("feeder_args", {}).get("setting", "dev")
        if self.split not in self.SPLIT_PREFIXES:
            print(f"[WARN] Unknown split '{self.split}', falling back to 'dev'")
            self.split = "dev"

        # Dummy id→name fallback
        self.dummy_map = {
            int(k): v["gloss"] for k, v in self.gloss_dict["id2gloss"].items()
        }

        # Model
        self.model = slr_network.TwoStream_Cosign(
            **self.cfg["model_args"], gloss_dict=self.gloss_dict
        )
        ckpt = torch.load(self.cfg["load_checkpoints"], map_location="cpu")
        sd = {}
        for k, v in ckpt["model_state_dict"].items():
            k = k[7:] if k.startswith("module.") else k
            sd[k] = v
        self.model.load_state_dict(sd, strict=False)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device).eval()
        self.norm_scale = self.cfg["model_args"]["norm_scale"]

        # Feeder — use split mode
        feeder_mode = self.split
        self.feeder = SkeletonFeeder(
            gloss_dict=self.gloss_dict,
            mode=feeder_mode,
            setting="si",
            transform_mode=False,
            datatype="skeleton",
            split=self.cfg["feeder_args"]["split"],
            norm_point=self.cfg["feeder_args"]["norm_point"],
            used_part=self.cfg["feeder_args"]["used_part"],
            skeleton_pkl=self.cfg["feeder_args"]["skeleton_pkl"],
        )
        self.id_to_idx = {
            item["video_id"]: i for i, item in enumerate(self.feeder.inputs_list)
        }
        print(f"[BoundaryExtractor] split={self.split}  samples={len(self.id_to_idx)}")

    def _collate(self, input_tensor: torch.Tensor):
        """Left/right pad a single sample and move to device."""
        T_orig = input_tensor.shape[0]
        lp = 6
        rp = int(np.ceil(T_orig / 4.0)) * 4 - T_orig + 6
        total_len = T_orig + lp + rp
        padded = torch.cat([
            input_tensor[:1].expand(lp, -1, -1),
            input_tensor,
            input_tensor[-1:].expand(total_len - T_orig - lp, -1, -1),
        ], dim=0)
        bx = padded.unsqueeze(0).to(self.device)
        bl = torch.LongTensor([int(np.ceil(T_orig / 4.0) * 4 + 12)]).to(self.device)
        return bx, bl, T_orig

    def __call__(self, sample_id: str) -> dict:
        """Extract boundaries for one sample.

        Args:
            sample_id: e.g. "02_0001"

        Returns:
            dict with sample_id, t_orig, feat_len, segments
        """
        if sample_id not in self.id_to_idx:
            raise KeyError(f"Sample '{sample_id}' not found in {self.split} split")

        idx = self.id_to_idx[sample_id]
        input_tensor, _, _ = self.feeder[idx]
        bx, bl, T_orig = self._collate(input_tensor)

        with torch.no_grad():
            ret = self.model({"x": bx, "len_x": bl})

        logits = ret["seq_logits"][0].cpu()
        feat_len = ret["feat_len"][0].item()

        from temposlr.utils.boundary_extraction import extract_boundaries, feat_to_orig

        raw_segments = extract_boundaries(logits, feat_len, norm_scale=self.norm_scale)

        segments = []
        for lid, sf, ef in raw_segments:
            so, eo = feat_to_orig(sf, ef, T_orig)
            gloss = self.dummy_map.get(lid, f"id_{lid}")
            segments.append({
                "gloss": gloss,
                "label_id": lid,
                "start": so,
                "end": eo,
                "start_feat": sf,
                "end_feat": ef,
            })

        return {
            "sample_id": sample_id,
            "t_orig": T_orig,
            "feat_len": feat_len,
            "segments": segments,
        }

    def batch(self, sample_ids: list) -> list:
        """Extract boundaries for multiple samples."""
        return [self(sid) for sid in sample_ids]

    def all_sample_ids(self) -> list:
        """Return all sample IDs for the current split, sorted."""
        ids = list(self.id_to_idx.keys())
        ids.sort()
        return ids


def _print_table(result: dict):
    """Pretty-print one result dict."""
    sid = result["sample_id"]
    t_orig = result["t_orig"]
    feat_len = result["feat_len"]
    segs = result["segments"]

    print(f"\n{'━' * 80}")
    print(f"  {sid}  |  {t_orig} frames  |  {feat_len} feat frames  |  {len(segs)} glosses")
    print(f"{'━' * 80}")

    if not segs:
        print("  (no glosses detected)")
        return

    print(f"  {'#':<4} {'Gloss':<25} {'ID':<8} {'Feat':<12} {'Orig':<12} {'Dur':<5}")
    print(f"  {'─' * 70}")
    for i, s in enumerate(segs):
        print(
            f"  {i+1:<4} {s['gloss']:<25} {s['label_id']:<8} "
            f"[{s['start_feat']:>3},{s['end_feat']:>3})  "
            f"[{s['start']:>3},{s['end']:>3})  "
            f"{s['end'] - s['start'] + 1:>3}"
        )


def main():
    parser = argparse.ArgumentParser(description="TEMPO gloss boundary extraction")
    parser.add_argument("samples", nargs="*", help="Sample IDs (e.g. 02_0001)")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument("--limit", type=int, default=10, help="Max samples with --all")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to use (default: dev)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="JSON output")
    parser.add_argument("--csv", action="store_true", dest="as_csv", help="CSV output")
    parser.add_argument("--config", default="configs/test.yaml", help="Config path")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    ext = BoundaryExtractor(config=args.config, device=args.device, split=args.split)

    # Determine sample IDs
    if args.all:
        all_ids = ext.all_sample_ids()
        sample_ids = all_ids[: args.limit]
    elif args.samples:
        sample_ids = args.samples
    else:
        parser.print_help()
        return

    results = ext.batch(sample_ids)

    # Output
    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.as_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["sample_id", "gloss", "label_id", "start", "end", "start_feat", "end_feat"])
        for r in results:
            for s in r["segments"]:
                writer.writerow([
                    r["sample_id"], s["gloss"], s["label_id"],
                    s["start"], s["end"], s["start_feat"], s["end_feat"],
                ])
    else:
        print(f"\n{'=' * 80}")
        print(f"GLOSS BOUNDARY EXTRACTION — {len(results)} samples ({args.split} split)")
        print(f"{'=' * 80}")
        for r in results:
            _print_table(r)
        print(f"\n{'=' * 80}")
        print(f"Total: {sum(len(r['segments']) for r in results)} glosses across {len(results)} samples")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
