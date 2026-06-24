"""
Video annotation overlay for sign language recognition.

Draws Arabic glosses on slowed-down video using ffmpeg drawtext filters.
Handles skeleton→video frame mapping, fps conversion, and ffmpeg pipeline.
"""

import os
import json
import subprocess


class VideoAnnotator:
    """Annotate sign language videos with Arabic gloss text overlays.

    Maps skeleton-level frame boundaries to video frame times, then
    drives ffmpeg drawtext filters to render each gloss on the slowed
    output video.
    """

    def __init__(
        self,
        font_path=None,
        font_size=36,
        font_color="white",
        border_width=3,
        border_color="black",
        fps_output=5,
        slow_factor=5,
        y_position="h-text_h-15",
    ):
        self.font_path = (
            font_path
            or "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"
        )
        self.font_size = font_size
        self.font_color = font_color
        self.border_width = border_width
        self.border_color = border_color
        self.fps_output = fps_output
        self.slow_factor = slow_factor
        self.y_position = y_position

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def probe_video(video_path: str) -> dict:
        """Run ffprobe and return video-stream metadata.

        Returns
            dict with keys: nb_frames, width, height, codec,
            duration, r_frame_rate, fps
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_frames,width,height,r_frame_rate,codec_name,duration",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = {}
        for line in result.stdout.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()

        parsed: dict = {
            "nb_frames": int(info.get("nb_frames", 0)),
            "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "codec": info.get("codec_name", ""),
            "duration": float(info.get("duration", 0)),
            "r_frame_rate": info.get("r_frame_rate", "0/1"),
        }
        # Parse rational frame rate → float
        rfr = parsed["r_frame_rate"]
        if "/" in rfr:
            num, den = rfr.split("/")
            parsed["fps"] = float(num) / float(den) if float(den) > 0 else 0.0
        else:
            parsed["fps"] = 0.0
        return parsed

    @staticmethod
    def skeleton_to_video_time(
        skeleton_frame: int,
        video_total_frames: int,
        skeleton_total_frames: int,
        fps_orig: float = 25.0,
        fps_output: float = 5.0,
    ) -> float:
        """Map a skeleton frame index to the output video time in seconds.

        The model's skeleton stream has fewer frames than the original
        video because ``TemporalRescale_test`` pads to a multiple of 4
        and takes every 2nd frame.  This method reverses the mapping so
        that drawtext ``between(t, …)`` uses correct output-timeline
        positions.

        Formula
            scale        = (video_frames - 1) / (skeleton_frames - 1)
            video_frame  = skeleton_frame * scale
            slow_factor  = fps_orig / fps_output
            time         = video_frame / fps_orig * slow_factor
        """
        if skeleton_total_frames <= 1:
            return 0.0
        scale = (video_total_frames - 1) / (skeleton_total_frames - 1)
        video_frame = skeleton_frame * scale
        slow_factor = fps_orig / fps_output
        return video_frame / fps_orig * slow_factor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """Escape characters special to ffmpeg ``drawtext`` filter syntax.

        This is used inside ``text='…'`` within a filter-chain argument.
        Single quotes and colons are the most common troublemakers.
        """
        return text.replace("'", "'\\''").replace(":", "\\:")

    def _build_drawtext_filters(
        self,
        segments: list,
        video_total_frames: int,
        skeleton_total_frames: int,
    ) -> list:
        """Build one ``drawtext`` filter string per segment.

        Each segment dict must have keys ``gloss``, ``start``, ``end``
        (inclusive skeleton-frame indices).  A small epsilon offset is
        added to non-first segments to prevent overlapping text when
        ffmpeg's ``between(t, …)`` includes both boundaries.
        """
        filters = []
        for i, s in enumerate(segments):
            t_start = self.skeleton_to_video_time(
                s["start"], video_total_frames, skeleton_total_frames,
            )
            t_end = self.skeleton_to_video_time(
                s["end"] + 1, video_total_frames, skeleton_total_frames,
            )
            # Tiny gap to avoid boundary overlap
            if i > 0:
                t_start += 0.001

            word_escaped = self._escape_drawtext(s["gloss"])
            filters.append(
                f"drawtext="
                f"fontfile={self.font_path}:"
                f"text='{word_escaped}':"
                f"fontsize={self.font_size}:"
                f"fontcolor={self.font_color}:"
                f"borderw={self.border_width}:"
                f"bordercolor={self.border_color}:"
                f"x=(w-text_w)/2:"
                f"y={self.y_position}:"
                f"enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'"
            )
        return filters

    # ------------------------------------------------------------------
    # Main annotation / ffmpeg driver
    # ------------------------------------------------------------------

    def annotate(
        self,
        video_path: str,
        segments: list,
        output_path: str,
        skeleton_total_frames: int = None,
    ) -> str:
        """Annotate *video_path* with gloss overlays and write to *output_path*.

        Strategy (tried in order):
            1. Single-pass — ``setpts`` + ``fps`` + drawtext filters.
            2. Two-pass — slow first, then overlay text.
            3. Fallback — just slow down without overlays.

        Parameters
            video_path:       input video file
            segments:         list of ``{gloss, start, end}`` dicts
            output_path:      where to write the result
            skeleton_total_frames:
                total number of skeleton frames (``t_orig`` from the
                model).  If *None*, inferred from ``max(end) + 1``.

        Returns
            *output_path* on success.
        """
        # ── probe input video ────────────────────────────────────────
        meta = self.probe_video(video_path)
        video_total_frames = meta["nb_frames"]

        if skeleton_total_frames is None:
            skeleton_total_frames = (
                max(s["end"] for s in segments) + 1 if segments else video_total_frames
            )

        # ── build drawtext filters ───────────────────────────────────
        filters = self._build_drawtext_filters(
            segments, video_total_frames, skeleton_total_frames,
        )

        slow_filter = f"setpts={self.slow_factor}*PTS,fps={self.fps_output}"

        if not filters:
            # No glosses → just slow down
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", slow_filter,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path

        # ── Pass 1: single-pass (slow + overlay) ─────────────────────
        filter_str = ",".join(filters)
        single_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"{slow_filter},{filter_str}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        r1 = subprocess.run(single_cmd, capture_output=True, text=True)
        if r1.returncode == 0:
            return output_path

        print(f"Single-pass ffmpeg failed, trying two-pass fallback …")
        print(f"  stderr (truncated): {r1.stderr[:400]}")

        # ── Pass 2: two-pass (slow first, then overlay) ──────────────
        base, ext = os.path.splitext(output_path)
        slow_path = f"{base}_slow{ext}"

        # 2a – slow
        cmd_slow = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", slow_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            slow_path,
        ]
        r2a = subprocess.run(cmd_slow, capture_output=True, text=True)
        if r2a.returncode != 0:
            print(f"Slow-only pass failed: {r2a.stderr[:300]}")
            return output_path

        # 2b – overlay text on slowed video
        cmd_overlay = [
            "ffmpeg", "-y",
            "-i", slow_path,
            "-vf", filter_str,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        r2b = subprocess.run(cmd_overlay, capture_output=True, text=True)
        if r2b.returncode != 0:
            print(f"Overlay pass failed, keeping slow-only version …")
            print(f"  stderr (truncated): {r2b.stderr[:300]}")
            os.replace(slow_path, output_path)
        else:
            os.remove(slow_path)

        return output_path

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def save_segments(
        segments: list,
        output_path: str,
        sample_id: str,
        total_skeleton_frames: int,
        fps_orig: int = 25,
        fps_output: int = 5,
    ) -> str:
        """Save segments as a JSON file with metadata and frame_word array.

        Returns *output_path*.
        """
        frame_words = [""] * total_skeleton_frames
        for s in segments:
            end = min(s["end"] + 1, total_skeleton_frames)
            for f in range(s["start"], end):
                frame_words[f] = s["gloss"]

        output_data = {
            "sample_id": sample_id,
            "total_frames": total_skeleton_frames,
            "fps_original": fps_orig,
            "fps_output": fps_output,
            "segments": segments,
            "frame_words": frame_words,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        return output_path
