"""
Video segmentation into individual sign clips based on gloss boundaries.

Takes a sign language video + skeleton data, runs the TEMPO model to detect
gloss boundaries, then cuts the video into one clip per detected sign.

Usage::

    # Segment a single video
    python -m temposlr.segment 00_1625 --split train

    # Multiple samples
    python -m temposlr.segment 00_1625 00_1626 --split train --output-dir ./segments

    # With custom config
    python -m temposlr.segment 00_1625 --config configs/test.yaml --slow 3
"""

import os
import sys
import json
import argparse
import subprocess

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from temposlr.extract import BoundaryExtractor
from temposlr.overlay import VideoAnnotator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def resolve_video_path(sample_id):
    """Find the video file for *sample_id* in common locations."""
    candidates = [
        f"{sample_id}.mp4",
        os.path.join("/kaggle/working", f"{sample_id}.mp4"),
        os.path.join("/kaggle/input", f"{sample_id}.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def skeleton_frame_to_video_time_sec(
    skeleton_frame: int,
    video_total_frames: int,
    skeleton_total_frames: int,
    fps_orig: float = 25.0,
) -> float:
    """Map skeleton frame index → original video time in seconds.

    Uses the same linear mapping as VideoAnnotator but WITHOUT the
    slow-down factor, since we want the *original* time boundaries
    for ffmpeg -ss / -to cuts.
    """
    if skeleton_total_frames <= 1:
        return 0.0
    scale = (video_total_frames - 1) / (skeleton_total_frames - 1)
    video_frame = skeleton_frame * scale
    return video_frame / fps_orig


def cut_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    slow_factor: int = 1,
    fps_output: int = 25,
) -> bool:
    """Cut a segment from *video_path* between *start_sec* and *end_sec*.

    If slow_factor > 1, the segment is slowed down by that factor.
    Returns True on success.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.4f}",
        "-i", video_path,
        "-t", f"{duration:.4f}",
    ]

    if slow_factor > 1:
        vf = f"setpts={slow_factor}*PTS,fps={fps_output}"
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# main segmentation
# ---------------------------------------------------------------------------

def segment_video(
    sample_id: str,
    extractor: BoundaryExtractor,
    video_path: str,
    output_dir: str,
    slow_factor: int = 1,
    fps_output: int = 25,
    fps_orig: float = 25.0,
    save_json: bool = True,
) -> dict:
    """Segment a video into per-sign clips.

    Parameters
    ----------
    sample_id : str
        e.g. "00_1625"
    extractor : BoundaryExtractor
        Pre-loaded extractor (avoids reloading model per sample).
    video_path : str
        Path to the input video.
    output_dir : str
        Where to write segment clips and JSON.
    slow_factor : int
        Slow-down factor (1 = original speed).
    fps_output : int
        Output fps when slowing down.
    fps_orig : float
        Original video fps.
    save_json : bool
        Whether to save the segment metadata JSON.

    Returns
    -------
    dict with sample_id, segments list, and output paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract boundaries
    print(f"[{sample_id}] Extracting gloss boundaries …")
    result = extractor(sample_id)
    segments = result["segments"]
    skeleton_total = result["t_orig"]
    print(f"  → {len(segments)} gloss(es) from {skeleton_total} skeleton frames")

    if not segments:
        print(f"  ⚠ no segments detected — skipping")
        return {"sample_id": sample_id, "segments": [], "clips": []}

    # 2. Probe video
    meta = VideoAnnotator.probe_video(video_path)
    video_frames = meta["nb_frames"]
    print(f"  Video: {meta['width']}×{meta['height']}, "
          f"{video_frames} frames @ {meta['fps']:.1f} fps, "
          f"{meta['duration']:.2f}s")

    # 3. Map boundaries to video time and cut
    clips = []
    for i, seg in enumerate(segments):
        t_start = skeleton_frame_to_video_time_sec(
            seg["start"], video_frames, skeleton_total, fps_orig,
        )
        # end + 1 because boundary is inclusive
        t_end = skeleton_frame_to_video_time_sec(
            seg["end"] + 1, video_frames, skeleton_total, fps_orig,
        )

        # Sanitise gloss for filename
        safe_gloss = seg["gloss"].replace(" ", "_").replace("/", "_")
        clip_name = f"{sample_id}_{i+1:03d}_{safe_gloss}.mp4"
        clip_path = os.path.join(output_dir, clip_name)

        print(f"  [{i+1}/{len(segments)}] {seg['gloss']:<20} "
              f"frames [{seg['start']:>3}, {seg['end']:>3}] "
              f"→ video [{t_start:.3f}s, {t_end:.3f}s]")

        ok = cut_video_segment(
            video_path, t_start, t_end, clip_path,
            slow_factor=slow_factor,
            fps_output=fps_output,
        )

        if ok and os.path.exists(clip_path):
            size_kb = os.path.getsize(clip_path) / 1024
            print(f"       ✓ {clip_name} ({size_kb:.0f} KB)")
            clips.append({
                "gloss": seg["gloss"],
                "label_id": seg["label_id"],
                "skeleton_start": seg["start"],
                "skeleton_end": seg["end"],
                "video_time_start": round(t_start, 4),
                "video_time_end": round(t_end, 4),
                "clip_file": clip_path,
            })
        else:
            print(f"       ✗ failed to cut segment")

    # 4. Optionally save metadata JSON
    if save_json:
        meta_path = os.path.join(output_dir, f"{sample_id}_segments.json")
        meta_data = {
            "sample_id": sample_id,
            "video_path": video_path,
            "skeleton_total_frames": skeleton_total,
            "video_total_frames": video_frames,
            "fps_original": fps_orig,
            "slow_factor": slow_factor,
            "num_segments": len(segments),
            "segments": clips,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ metadata → {meta_path}")

    return {
        "sample_id": sample_id,
        "segments": segments,
        "clips": clips,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python -m temposlr.segment",
        description="Segment sign language video into per-sign clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m temposlr.segment 00_1625 --split train
  python -m temposlr.segment 00_1625 00_1626 --split train --slow 3
  python -m temposlr.segment 00_1625 --config configs/test.yaml --no-json
""",
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
        "--output-dir", default="/kaggle/working/output/segments",
        help="Directory for output clips (default: /kaggle/working/output/segments)",
    )
    parser.add_argument(
        "--slow", type=int, default=1,
        help="Slow-down factor for clips (default: 1 = original speed)",
    )
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Output fps when slowing down (default: 25)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip saving per-sample JSON metadata",
    )
    args = parser.parse_args()

    if not args.samples:
        parser.print_help()
        print("\n[ERROR] At least one sample ID is required.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once, segment all samples
    print("[setup] Loading BoundaryExtractor …")
    ext = BoundaryExtractor(config=args.config, split=args.split)

    results = []
    for sample_id in args.samples:
        print(f"\n{'=' * 70}")
        video_path = resolve_video_path(sample_id)
        if video_path is None:
            print(f"[{sample_id}] ✗ no video found — skipping")
            continue

        res = segment_video(
            sample_id=sample_id,
            extractor=ext,
            video_path=video_path,
            output_dir=args.output_dir,
            slow_factor=args.slow,
            fps_output=args.fps,
            save_json=not args.no_json,
        )
        results.append(res)

    # Summary
    print(f"\n{'=' * 70}")
    print("SEGMENTATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Sample':<12} {'Glosses':<10} {'Clips':<10} {'Output Dir'}")
    print(f"  {'─' * 60}")
    for r in results:
        print(f"  {r['sample_id']:<12} {len(r['segments']):<10} "
              f"{len(r['clips']):<10} {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
