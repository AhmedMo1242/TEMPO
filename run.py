#!/usr/bin/env python3
"""
TEMPO Unified CLI — Train, Evaluate, Extract, Predict.

Usage:
    python run.py train --config configs/train.yaml
    python run.py eval  --config configs/test.yaml --split dev
    python run.py extract --split dev --all --limit 10
    python run.py extract 02_0001 02_0002
    python run.py predict 02_0001 --json
"""
import argparse
import sys
import os

# Ensure TEMPO root is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def cmd_train(args):
    """Run training."""
    import yaml
    from temposlr.runner import SLRProcessor

    with open(args.config, "r") as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.set_defaults(**default_arg)
    p = parser.parse_args(args.extra_args or [])

    proc = SLRProcessor(p)
    proc.start()


def cmd_eval(args):
    """Run evaluation on dev or test split."""
    import yaml
    from temposlr.runner import SLRProcessor

    with open(args.config, "r") as f:
        default_arg = yaml.load(f, Loader=yaml.FullLoader)

    # Override phase to test
    default_arg["phase"] = "test"
    if args.split:
        default_arg["feeder_args"]["setting"] = args.split

    parser = argparse.ArgumentParser()
    parser.set_defaults(**default_arg)
    p = parser.parse_args(args.extra_args or [])

    proc = SLRProcessor(p)
    proc.start()


def cmd_extract(args):
    """Extract gloss boundaries from skeleton data."""
    from temposlr.extract import BoundaryExtractor, _print_table

    ext = BoundaryExtractor(
        config=args.config,
        device=args.device,
        split=args.split,
    )

    if args.all:
        all_ids = ext.all_sample_ids()
        sample_ids = all_ids[: args.limit]
    elif args.samples:
        sample_ids = args.samples
    else:
        print("Provide sample IDs or use --all")
        return

    results = ext.batch(sample_ids)

    if args.as_json:
        import json
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.as_csv:
        import csv
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
        print(f"\nTotal: {sum(len(r['segments']) for r in results)} glosses across {len(results)} samples")


def cmd_predict(args):
    """Run model prediction (inference) and print decoded glosses."""
    from temposlr.extract import BoundaryExtractor

    ext = BoundaryExtractor(
        config=args.config,
        device=args.device,
        split=args.split,
    )

    if args.all:
        all_ids = ext.all_sample_ids()
        sample_ids = all_ids[: args.limit]
    elif args.samples:
        sample_ids = args.samples
    else:
        print("Provide sample IDs or use --all")
        return

    results = ext.batch(sample_ids)

    if args.as_json:
        import json
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'─' * 60}")
        print(f"  Predictions ({args.split} split) — {len(results)} samples")
        print(f"{'─' * 60}")
        for r in results:
            glosses = " ".join(s["gloss"] for s in r["segments"])
            print(f"  {r['sample_id']}: {glosses}")
        print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="TEMPO Sign Language Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py train --config configs/train.yaml
    python run.py eval --config configs/test.yaml
    python run.py extract --split dev --all --limit 10
    python run.py extract 02_0001 02_0002 02_0003
    python run.py predict --split dev --all --limit 5
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # --- train ---
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--config", default="configs/train.yaml", help="Config path")
    p_train.add_argument("extra_args", nargs="*", help="Extra args passed to config")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Evaluate model on dev/test")
    p_eval.add_argument("--config", default="configs/test.yaml", help="Config path")
    p_eval.add_argument("--split", choices=["dev", "test"], default="dev", help="Split")
    p_eval.add_argument("extra_args", nargs="*", help="Extra args passed to config")

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract gloss boundaries")
    p_ext.add_argument("samples", nargs="*", help="Sample IDs (e.g. 02_0001)")
    p_ext.add_argument("--all", action="store_true", help="Process all samples")
    p_ext.add_argument("--limit", type=int, default=10, help="Max samples with --all")
    p_ext.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p_ext.add_argument("--config", default="configs/test.yaml", help="Config path")
    p_ext.add_argument("--device", default=None, help="cuda or cpu")
    p_ext.add_argument("--json", action="store_true", dest="as_json")
    p_ext.add_argument("--csv", action="store_true", dest="as_csv")

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Predict glosses (no boundaries)")
    p_pred.add_argument("samples", nargs="*", help="Sample IDs")
    p_pred.add_argument("--all", action="store_true")
    p_pred.add_argument("--limit", type=int, default=10)
    p_pred.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    p_pred.add_argument("--config", default="configs/test.yaml")
    p_pred.add_argument("--device", default=None)
    p_pred.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
