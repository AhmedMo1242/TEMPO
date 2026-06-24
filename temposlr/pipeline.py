"""
Pipeline: extract Arabic glosses from sign language video → annotate with text overlay.

Usage::

    python -m temposlr.pipeline 00_1625 --split train
    python -m temposlr.pipeline 02_0001 02_0002 --split dev --output-dir ./output
    python -m temposlr.pipeline 00_1625 --split train --config configs/test.yaml --fps 5
"""

import os
import sys
import argparse
import textwrap

# Ensure the package root is on sys.path so that ``temposlr`` is
# importable regardless of current working directory.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from temposlr.extract import BoundaryExtractor
from temposlr.overlay import VideoAnnotator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_video_path(sample_id):
    """Find the video file for *sample_id* in common locations.

    Checks, in order:
        1. current working directory
        2. ``/kaggle/working/``
        3. ``/kaggle/input/``
    """
    candidates = [
        f"{sample_id}.mp4",
        os.path.join("/kaggle/working", f"{sample_id}.mp4"),
        os.path.join("/kaggle/input", f"{sample_id}.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python -m temposlr.pipeline",
        description="TEMPO sign language video annotation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:

              python -m temposlr.pipeline 00_1625 --split train
              python -m temposlr.pipeline 02_0001 02_0002 --split dev \\
                      --output-dir ./output
              python -m temposlr.pipeline 00_1625 --split train \\
                      --config configs/test.yaml --fps 5
        """),
    )
    parser.add_argument(
        "samples", nargs="*",
        help="Sample ID(s) to process (e.g. 00_1625)",
    )
    parser.add_argument(
        "--split", choices=["train", "dev", "test"],
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--config", default="configs/test.yaml",
        help="Path to model config YAML (default: configs/test.yaml)",
    )
    parser.add_argument(
        "--output-dir", default="/kaggle/working/output",
        help="Directory for output files (default: /kaggle/working/output)",
    )
    parser.add_argument(
        "--fps", type=int, default=5,
        help="Output video frame rate (default: 5)",
    )
    parser.add_argument(
        "--font-path", default=None,
        help="Path to a TTF font that supports Arabic script "
             "(default: NotoSansArabic-Bold)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Save per-sample segments JSON files",
    )
    args = parser.parse_args()

    if not args.samples:
        parser.print_help()
        print("\n[ERROR] At least one sample ID is required.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Create the model-driven boundary extractor
    # ------------------------------------------------------------------
    print("[setup] Initialising BoundaryExtractor …")
    ext = BoundaryExtractor(config=args.config, split=args.split)

    # ------------------------------------------------------------------
    # 2.  Create the video annotator
    # ------------------------------------------------------------------
    font_path = (
        args.font_path
        or "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"
    )
    annotator = VideoAnnotator(font_path=font_path, fps_output=args.fps)

    # ------------------------------------------------------------------
    # 3.  Process each sample
    # ------------------------------------------------------------------
    results = []

    for sample_id in args.samples:
        print(f"\n{'=' * 70}")
        print(f"Sample: {sample_id}")
        print(f"{'=' * 70}")

        # 3a – extract gloss boundaries from skeleton data
        print("[1/4] Extracting gloss boundaries …")
        result = ext(sample_id)
        segments = result["segments"]
        skeleton_total = result["t_orig"]
        print(
            f"  → {len(segments)} segment(s) across "
            f"{skeleton_total} skeleton frames"
        )

        if segments:
            print(f"  {'#':<4} {'Gloss':<25} {'Frame Range':<15} {'Duration':<8}")
            print(f"  {'─' * 55}")
            for i, s in enumerate(segments):
                dur = s["end"] - s["start"] + 1
                print(
                    f"  {i+1:<4} {s['gloss']:<25} "
                    f"[{s['start']:>3}, {s['end']:>3}]  {dur:>3} frames"
                )

        # 3b – locate the video file
        print("[2/4] Locating video file …")
        video_path = resolve_video_path(sample_id)
        if video_path is None:
            print(f"  ✗ no video found for sample '{sample_id}' — skipping")
            continue
        print(f"  ✓ {video_path}")

        meta = VideoAnnotator.probe_video(video_path)
        print(
            f"    {meta['width']}×{meta['height']}, "
            f"{meta['nb_frames']} frames @ {meta['fps']:.1f} fps, "
            f"{meta['duration']:.2f}s"
        )

        # 3c – annotate / overlay
        print("[3/4] Running annotation …")
        output_video = os.path.join(
            args.output_dir, f"{sample_id}_annotated.mp4"
        )
        annotator.annotate(
            video_path,
            segments,
            output_video,
            skeleton_total_frames=skeleton_total,
        )

        if os.path.exists(output_video):
            size_kb = os.path.getsize(output_video) / 1024
            print(f"  ✓ output → {output_video} ({size_kb:.0f} KB)")

        # 3d – optional: save JSON
        if args.json:
            seg_path = os.path.join(
                args.output_dir, f"{sample_id}_segments.json"
            )
            VideoAnnotator.save_segments(
                segments, seg_path, sample_id, skeleton_total,
            )
            print(f"  ✓ segments → {seg_path}")

        results.append(
            {
                "sample_id": sample_id,
                "segments": len(segments),
                "video": video_path,
                "output": output_video if os.path.exists(output_video) else None,
            }
        )

    # ------------------------------------------------------------------
    # 4.  Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Sample':<12} {'Glosses':<10} {'Output':<40}")
    print(f"  {'─' * 62}")
    for r in results:
        label = r["output"] or "FAILED"
        print(f"  {r['sample_id']:<12} {r['segments']:<10} {label:<40}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
