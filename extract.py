#!/usr/bin/env python3
"""
Gloss boundary extraction from TEMPO skeleton inference.

CLI:
    python extract.py 02_0001                    # one sample
    python extract.py 02_0001 02_0002 02_0003    # multiple
    python extract.py --all --limit 10            # first 10 from dataset
    python extract.py 02_0001 --json             # JSON output
    python extract.py 02_0001 --csv              # CSV output

Library:
    from extract import BoundaryExtractor
    ext = BoundaryExtractor()
    segments = ext("02_0001")
    # [{"gloss": "هو", "label_id": 1071, "start": 0, "end": 5, "start_feat": 2, "end_feat": 3}, ...]
"""
import os, sys, json, yaml, csv, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BoundaryExtractor:
    """Extract gloss boundaries from TEMPO model inference.

    Usage:
        ext = BoundaryExtractor(config="configs/test.yaml")
        segments = ext("02_0001")
    """

    def __init__(
        self,
        config: str = "configs/test.yaml",
        gloss_dict_path: str = "./datasets/mslr2026/si_gloss_dict.json",
        device: str = None,
    ):
        """Load model and build dataset feeder.

        Args:
            config: path to configs/test.yaml
            gloss_dict_path: path to gloss dict JSON
            device: "cuda" or "cpu" (auto-detected if None)
        """
        from datasets.skeleton_feeder import SkeletonFeeder
        import slr_network

        with open(config) as f:
            self.cfg = yaml.safe_load(f)
        with open(gloss_dict_path) as f:
            self.gloss_dict = json.load(f)

        # Dummy id→name fallback
        self.dummy_map = {int(k): v["gloss"] for k, v in self.gloss_dict["id2gloss"].items()}

        # Model
        self.model = slr_network.TwoStream_Cosign(
            **self.cfg["model_args"], gloss_dict=self.gloss_dict
        )
        ckpt = torch.load(self.cfg["load_checkpoints"], map_location="cpu")
        sd = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in ckpt["model_state_dict"].items()
        }
        self.model.load_state_dict(sd, strict=True)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device).eval()
        self.norm_scale = self.cfg["model_args"]["norm_scale"]

        # Feeder
        self.feeder = SkeletonFeeder(
            gloss_dict=self.gloss_dict,
            mode="test",
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

    def __call__(
        self,
        sample_id: str,
        gloss_map: dict = None,
    ) -> dict:
        """Extract boundaries for one sample.

        Args:
            sample_id: e.g. "02_0001"
            gloss_map: optional {label_id: "arabic_gloss"} mapping

        Returns:
            {
                "sample_id": str,
                "t_orig": int,
                "feat_len": int,
                "segments": [
                    {"gloss": str, "label_id": int,
                     "start": int, "end": int,        # original frames (inclusive)
                     "start_feat": int, "end_feat": int  # feature frames
                    }, ...
                ]
            }
        """
        if sample_id not in self.id_to_idx:
            raise KeyError(f"Sample '{sample_id}' not found in dataset")

        idx = self.id_to_idx[sample_id]
        input_tensor, _, _ = self.feeder[idx]
        bx, bl, T_orig = self._collate(input_tensor)

        with torch.no_grad():
            ret = self.model({"x": bx, "len_x": bl})

        logits = ret["seq_logits"][0].cpu()
        feat_len = ret["feat_len"][0].item()

        from utils.boundary_extraction import extract_boundaries, feat_to_orig

        raw_segments = extract_boundaries(logits, feat_len, norm_scale=self.norm_scale)

        segments = []
        for lid, sf, ef in raw_segments:
            so, eo = feat_to_orig(sf, ef, T_orig)
            gloss = None
            if gloss_map and lid in gloss_map:
                gloss = gloss_map[lid]
            elif lid in self.dummy_map:
                gloss = self.dummy_map[lid]
            else:
                gloss = f"id_{lid}"
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

    def batch(
        self,
        sample_ids: list,
        gloss_map: dict = None,
    ) -> list:
        """Extract boundaries for multiple samples.

        Args:
            sample_ids: list of sample IDs
            gloss_map: optional {label_id: "arabic_gloss"} mapping

        Returns:
            list of result dicts (one per sample)
        """
        return [self(sid, gloss_map=gloss_map) for sid in sample_ids]

    def build_gloss_map_from_csv(
        self,
        csv_path: str = "/kaggle/input/mslr-track-2-5/best_dev_seq.csv",
        max_samples: int = 50,
    ) -> dict:
        """Build real {label_id: gloss_name} mapping from CSV predictions.

        Matches model-decoded label_ids to CSV tokens 1:1.
        """
        from utils.boundary_extraction import extract_boundaries

        csv_preds = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                csv_preds[row["id"]] = row["gloss"]

        id2gloss = {}
        count = 0
        for sample_id, pred_text in csv_preds.items():
            if sample_id not in self.id_to_idx:
                continue
            result = self(sample_id)
            decoded_ids = [s["label_id"] for s in result["segments"]]
            csv_tokens = pred_text.split()

            if len(decoded_ids) == len(csv_tokens):
                for lid, token in zip(decoded_ids, csv_tokens):
                    if lid not in id2gloss:
                        id2gloss[lid] = token
                count += 1
                if count >= max_samples:
                    break
        return id2gloss


def _print_table(result: dict, gloss_map: dict = None):
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
        g = s["gloss"]
        if gloss_map and s["label_id"] in gloss_map:
            g = gloss_map[s["label_id"]]
        print(
            f"  {i+1:<4} {g:<25} {s['label_id']:<8} "
            f"[{s['start_feat']:>3},{s['end_feat']:>3})  "
            f"[{s['start']:>3},{s['end']:>3})  "
            f"{s['end'] - s['start'] + 1:>3}"
        )


def main():
    parser = argparse.ArgumentParser(description="TEMPO gloss boundary extraction")
    parser.add_argument("samples", nargs="*", help="Sample IDs (e.g. 02_0001)")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument("--limit", type=int, default=10, help="Max samples with --all")
    parser.add_argument("--json", action="store_true", dest="as_json", help="JSON output")
    parser.add_argument("--csv", action="store_true", dest="as_csv", help="CSV output")
    parser.add_argument("--config", default="configs/test.yaml", help="Config path")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    ext = BoundaryExtractor(config=args.config, device=args.device)

    # Determine sample IDs
    if args.all:
        all_ids = list(ext.id_to_idx.keys())
        all_ids.sort()
        sample_ids = all_ids[: args.limit]
    elif args.samples:
        sample_ids = args.samples
    else:
        parser.print_help()
        return

    # Build gloss map from CSV if available
    gloss_map = {}
    csv_path = "/kaggle/input/mslr-track-2-5/best_dev_seq.csv"
    if os.path.exists(csv_path):
        gloss_map = ext.build_gloss_map_from_csv(csv_path)
        print(f"Loaded {len(gloss_map)} gloss mappings from CSV")

    # Process
    results = ext.batch(sample_ids, gloss_map=gloss_map)

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
        print(f"GLOSS BOUNDARY EXTRACTION — {len(results)} samples")
        print(f"{'=' * 80}")
        for r in results:
            _print_table(r, gloss_map)
        print(f"\n{'=' * 80}")
        print(f"Total: {sum(len(r['segments']) for r in results)} glosses across {len(results)} samples")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
