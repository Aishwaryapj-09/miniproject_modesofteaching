"""
analyze_video.py
================
Main Analysis Pipeline — Pre-recorded Video Processing
Project: A Unified Behavioural Intelligence System for Academic Spaces

SCOPE (as per revised requirements):
  - Processes PRE-RECORDED video files (not live camera capture)
  - No live video capture / webcam support (as requested)
  - No YOLOv8 face detection (as requested; uses Haar cascade + MediaPipe)
  - Frame-wise per-frame logging to CSV + terminal
  - Annotated output video saved with overlays
  - Sample frames saved to test_frames/

DECISION LOGIC (matches design doc):
  Face ENTHUSIASTIC (happy/surprise)   → ENTHUSIASTIC
  Face NOT enthusiastic                → NOT ENTHUSIASTIC
  No face + body ACTIVE (smoothed ≥ 0.08) → ENTHUSIASTIC  (side/back cam)
  No face + body still                 → NOT ENTHUSIASTIC

OUTPUT STRUCTURE:
  outputs/
    videos/   analyzed_<name>_<timestamp>.mp4
    frames/   <name>_<timestamp>/  frame_000030_enthusiastic.jpg …
    logs/     <name>_<timestamp>_frame_log.csv

USAGE:
    python src/analyze_video.py --video path/to/video.mp4 --model models/face_model.keras
    python src/analyze_video.py --video path/to/video.mp4 --model models/face_model.keras \\
        --fps_sample 5 --save_csv

    # Batch: analyse all videos in a folder
    python src/analyze_video.py --video_dir data/test/enthusiastic --model models/face_model.keras
"""

import argparse, os, sys, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Local modules (same src/ directory)
sys.path.insert(0, str(Path(__file__).parent))
from face_expression import FaceExpressionClassifier
from body_language   import BodyLanguageAnalyser

# ─────────────────────────────────────────────────────────────────────────────
#  Output roots
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path("outputs")

# ── Visual colours (BGR) ─────────────────────────────────────────────────────
COLOUR_ENT     = (0,  220,  80)
COLOUR_NOT     = (0,   60, 220)
COLOUR_TEXT    = (255, 255, 255)
COLOUR_OVERLAY = (30,  30,  30)
COLOUR_YELLOW  = (0,  220, 220)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Float seconds → 'MM:SS'."""
    seconds = max(0.0, float(seconds))
    m, s    = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Decision logic
# ─────────────────────────────────────────────────────────────────────────────

def decide_frame(face_result: dict, body_result: dict) -> tuple[bool, str]:
    """
    Combines face and body signals into a single per-frame decision.

    Primary: face expression (happy/surprise → enthusiastic).
    Fallback: body movement (when face not visible — side camera or back view).

    Returns:
        (is_enthusiastic: bool, reason: str)
    """
    face_found  = face_result["face_found"]
    face_enth   = face_result["is_enthusiastic"]
    prob        = face_result["probability"]
    body_active = body_result["is_enthusiastic"]
    body_score  = body_result["smoothed_score"]

    if face_found:
        if face_enth:
            reason = (f"face=ENTHUS (prob={prob:.2f}) "
                      f"body={'active' if body_active else 'still'}")
            return True, reason
        else:
            reason = (f"face=NOT_enthus (prob={prob:.2f}) "
                      f"body={'active(ignored)' if body_active else 'still'}")
            return False, reason
    else:
        # No face visible — fall back to body movement
        if body_result["detected"] and body_active:
            reason = f"no_face(side/back_cam) body=ACTIVE (score={body_score:.3f})"
            return True, reason
        else:
            reason = f"no_face(side/back_cam) body=still (score={body_score:.3f})"
            return False, reason


# ─────────────────────────────────────────────────────────────────────────────
#  Frame overlay drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, frame_idx: int, ts: float,
                 face_result: dict, body_result: dict,
                 frame_label: str, reason: str,
                 enth_count: int, total_count: int) -> np.ndarray:
    """Draw all information overlays onto an annotated frame."""
    h, w = frame.shape[:2]

    # Semi-transparent dark header bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 120), COLOUR_OVERLAY, -1)
    frame = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)

    col_label = COLOUR_ENT if frame_label == "enthusiastic" else COLOUR_NOT

    # Timestamp + frame number
    m_ts, s_ts = divmod(int(ts), 60)
    cv2.putText(frame, f"[{m_ts:02d}:{s_ts:02d}] Frame {frame_idx}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_TEXT, 1)

    # Main label
    cv2.putText(frame, frame_label.upper(),
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_label, 2)

    # Reason string (truncated)
    cv2.putText(frame, reason[:75],
                (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOUR_TEXT, 1)

    # Face info line
    if face_result["face_found"]:
        fc = COLOUR_ENT if face_result["is_enthusiastic"] else COLOUR_NOT
        fs = (f"FACE: {'ENTHUS' if face_result['is_enthusiastic'] else 'NOT enthus'} "
              f"| prob={face_result['probability']:.2f}")
    else:
        fc, fs = COLOUR_YELLOW, "FACE: not detected (using body fallback)"
    cv2.putText(frame, fs, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.36, fc, 1)

    # Body info line
    if body_result["detected"]:
        bc = COLOUR_ENT if body_result["is_enthusiastic"] else COLOUR_NOT
        bs = (f"BODY: {'ACTIVE' if body_result['is_enthusiastic'] else 'still'} "
              f"| score={body_result['smoothed_score']:.3f}")
    else:
        bc, bs = COLOUR_YELLOW, "BODY: not detected"
    cv2.putText(frame, bs, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.36, bc, 1)

    # Running tally (top-right)
    pct = int(100 * enth_count / max(total_count, 1))
    cv2.putText(frame, f"Enth: {enth_count}/{total_count} ({pct}%)",
                (w - 240, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_TEXT, 1)

    # Coloured right-edge stripe
    cv2.rectangle(frame, (w - 12, 0), (w, h), col_label, -1)

    # Movement bar (bottom-right)
    bx, bh_bar, by, bw2 = w - 40, 120, h - 145, 18
    filled = int(bh_bar * min(body_result["smoothed_score"] / 0.3, 1.0))
    cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh_bar), (60, 60, 60), -1)
    if filled > 0:
        bar_c = COLOUR_ENT if body_result["smoothed_score"] >= 0.08 else COLOUR_NOT
        cv2.rectangle(frame,
                      (bx, by + bh_bar - filled),
                      (bx + bw2, by + bh_bar), bar_c, -1)
    cv2.putText(frame, "MOV", (bx - 2, by - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOUR_TEXT, 1)
    cv2.putText(frame, f"{body_result['smoothed_score']:.2f}",
                (bx - 4, by + bh_bar + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOUR_TEXT, 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Core analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyse_video(video_path: str, model_path: str,
                  fps_sample: int = 30,
                  save_video: bool = True,
                  movement_threshold: float = 0.08,
                  face_threshold: float = 0.5) -> dict:
    """
    Analyse a single pre-recorded video file, frame by frame.

    Args:
        video_path          : Path to input .mp4 / .avi / .mov
        model_path          : Path to trained face_model.keras
        fps_sample          : Analyse 1 in every N frames (default 30)
        save_video          : Write annotated output video
        movement_threshold  : Body movement score threshold
        face_threshold      : Face enthusiasm probability threshold

    Returns:
        dict with analysis results, logs, and output file paths.
    """
    if not os.path.isfile(video_path):
        sys.exit(f"[ERROR] Video not found: {video_path}")

    # ── Setup output folders ──────────────────────────────────────────────
    vid_stem = Path(video_path).stem
    run_ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dir_videos = OUTPUT_ROOT / "videos"
    dir_frames = OUTPUT_ROOT / "frames" / f"{vid_stem}_{run_ts}"
    dir_logs   = OUTPUT_ROOT / "logs"

    for d in [dir_videos, dir_frames, dir_logs]:
        d.mkdir(parents=True, exist_ok=True)

    out_video_path = str(dir_videos / f"analyzed_{vid_stem}_{run_ts}.mp4")
    out_csv_path   = str(dir_logs   / f"{vid_stem}_{run_ts}_frame_log.csv")

    # ── Load models ───────────────────────────────────────────────────────
    face_clf = FaceExpressionClassifier(model_path, conf_threshold=face_threshold)
    body_an  = BodyLanguageAnalyser(movement_threshold=movement_threshold)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    orig_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    true_duration = total_frames / max(orig_fps, 1.0)

    print(f"\n{'='*62}")
    print(f"  Video   : {Path(video_path).name}")
    print(f"  Size    : {width}×{height}  FPS={orig_fps:.1f}  "
          f"Frames={total_frames}  Duration={fmt_time(true_duration)}")
    print(f"  Sampling: 1 in every {fps_sample} frames")
    print(f"\n  DECISION LOGIC:")
    print(f"    Face happy/surprise → ENTHUSIASTIC")
    print(f"    Face angry/neutral  → NOT ENTHUSIASTIC")
    print(f"    No face + body active (score≥{movement_threshold}) → ENTHUSIASTIC (fallback)")
    print(f"\n  Output video  : {out_video_path}")
    print(f"  Output frames : {dir_frames}")
    print(f"  Log CSV       : {out_csv_path}")
    print(f"{'='*62}\n")

    # ── Video writer ──────────────────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = max(orig_fps / fps_sample, 1.0)
        writer  = cv2.VideoWriter(out_video_path, fourcc, out_fps, (width, height))

    # ── Frame loop ────────────────────────────────────────────────────────
    all_predictions = []   # list of (label, timestamp_seconds)
    logs            = []   # per-frame log dicts
    enth_count      = 0
    not_enth_count  = 0
    analysed_i      = 0
    saved_frames    = 0
    frame_i         = 0

    body_an.reset()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % fps_sample == 0:
            ts = frame_i / max(orig_fps, 1.0)

            # ── Analysis ──────────────────────────────────────────────────
            body_result = body_an.analyse_frame(frame)
            face_result = face_clf.analyse_frame(frame)

            frame_enth, reason = decide_frame(face_result, body_result)
            frame_label        = "enthusiastic" if frame_enth else "not enthusiastic"

            if frame_enth:
                enth_count += 1
            else:
                not_enth_count += 1
            analysed_i += 1

            all_predictions.append((frame_label, ts))

            # ── Build annotated frame ──────────────────────────────────────
            # Start from body-annotated frame (has skeleton overlay)
            annotated = body_result["annotated_frame"].copy()

            # Add face bounding box on top
            if face_result["face_found"]:
                x, y, w2, h2 = face_result["face_bbox"]
                col = COLOUR_ENT if frame_enth else COLOUR_NOT
                cv2.rectangle(annotated, (x, y), (x + w2, y + h2), col, 2)

            # Add info overlay header
            annotated = draw_overlay(
                annotated, frame_i, ts,
                face_result, body_result,
                frame_label, reason,
                enth_count, analysed_i
            )

            # ── Save frame image ───────────────────────────────────────────
            img_name = (f"frame_{frame_i:06d}_"
                        f"{frame_label.replace(' ', '_')}.jpg")
            cv2.imwrite(str(dir_frames / img_name), annotated)
            saved_frames += 1

            if writer:
                writer.write(annotated)

            # ── Terminal output ────────────────────────────────────────────
            m_ts, s_ts = divmod(int(ts), 60)
            face_info  = (f"prob={face_result['probability']:.2f}"
                          if face_result["face_found"] else "no_face→body")
            body_info  = (f"body={'active' if body_result['is_enthusiastic'] else 'still'}"
                          f"({body_result['smoothed_score']:.3f})")
            print(f"  [{m_ts:02d}:{s_ts:02d}] frame {frame_i:05d} "
                  f"| {frame_label.upper():<16s} "
                  f"| {face_info}  {body_info}")

            # ── Log entry ──────────────────────────────────────────────────
            logs.append({
                "frame_index":      frame_i,
                "timestamp_sec":    round(ts, 3),
                "timestamp_mmss":   f"{m_ts:02d}:{s_ts:02d}",
                "face_found":       face_result["face_found"],
                "face_probability": round(face_result["probability"], 4),
                "face_enthusiastic":face_result["is_enthusiastic"],
                "body_detected":    body_result["detected"],
                "body_score":       round(body_result["smoothed_score"], 4),
                "body_enthusiastic":body_result["is_enthusiastic"],
                "decision_reason":  reason,
                "frame_label":      frame_label,
            })

        frame_i += 1

    # ── Cleanup ────────────────────────────────────────────────────────────
    cap.release()
    body_an.close()
    if writer:
        writer.release()

    # ── Save CSV log ───────────────────────────────────────────────────────
    if logs:
        pd.DataFrame(logs).to_csv(out_csv_path, index=False)
        print(f"\n[LOG] Frame log saved → {out_csv_path}")

    return {
        "all_predictions":         all_predictions,
        "enthusiastic_frames":     enth_count,
        "not_enthusiastic_frames": not_enth_count,
        "total_analysed_frames":   analysed_i,
        "true_duration":           true_duration,
        "fps":                     orig_fps,
        "frame_log":               logs,
        "output_video":            out_video_path if save_video else None,
        "frames_dir":              str(dir_frames),
        "log_csv":                 out_csv_path,
        "saved_images":            saved_frames,
        "total_raw_frames":        frame_i,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, video_path: str) -> None:
    """Print final summary report to terminal."""
    preds     = result["all_predictions"]
    fps       = result["fps"]
    frame_dur = 1.0 / max(fps, 1.0)

    # Compute time spent per label using inter-frame gaps
    time_per = {"enthusiastic": 0.0, "not enthusiastic": 0.0}
    for i, (lbl, ts) in enumerate(preds):
        gap = (preds[i + 1][1] - ts) if i + 1 < len(preds) else frame_dur
        if gap <= 0:
            gap = frame_dur
        time_per[lbl] = time_per.get(lbl, 0.0) + gap

    total_time = sum(time_per.values())
    t_enth     = time_per.get("enthusiastic",     0.0)
    t_not      = time_per.get("not enthusiastic", 0.0)
    pct_enth   = t_enth / total_time * 100 if total_time > 0 else 0.0
    pct_not    = t_not  / total_time * 100 if total_time > 0 else 0.0

    # Verdict (by time, not frame count)
    if t_enth >= t_not:
        verdict      = "TEACHER IS TEACHING ENTHUSIASTICALLY"
        verdict_col  = "Enthusiastic"
    else:
        verdict      = "TEACHER IS NOT TEACHING ENTHUSIASTICALLY"
        verdict_col  = "Not Enthusiastic"

    sep = "=" * 62
    print(f"\n{sep}")
    print("  TEACHER ENTHUSIASM ANALYSIS — FINAL REPORT")
    print(sep)

    dur = result["true_duration"] if result["true_duration"] > 0 else total_time
    print(f"\n  Video            : {Path(video_path).name}")
    print(f"  Video duration   : {fmt_time(dur)}")
    print(f"  Total frames     : {result['total_raw_frames']}")
    print(f"  Frames analysed  : {result['total_analysed_frames']}")
    print(f"  Frame images     : {result['saved_images']}")

    print(f"\n  {'─'*54}")
    print(f"  Time-based breakdown:")
    print(f"  {'─'*54}")

    bar_e = "█" * int(pct_enth / 5)
    bar_n = "█" * int(pct_not  / 5)
    print(f"  {'Enthusiastic':<20s} {fmt_time(t_enth)} ({pct_enth:5.1f}%)  {bar_e}")
    print(f"  {'Not enthusiastic':<20s} {fmt_time(t_not)}  ({pct_not:5.1f}%)  {bar_n}")

    print(f"\n  {'─'*54}")
    print(f"  DOMINANT STATE   : {verdict_col}")
    print(f"  VERDICT          : {verdict}")
    print(f"  {'─'*54}")

    if result["output_video"]:
        print(f"\n  Annotated video  : {result['output_video']}")
    print(f"  Saved frames at  : {result['frames_dir']}")
    print(f"  Frame log CSV    : {result['log_csv']}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Batch mode helper
# ─────────────────────────────────────────────────────────────────────────────

def analyse_directory(video_dir: str, args) -> None:
    """Analyse all video files in a directory."""
    video_dir = Path(video_dir)
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in extensions]

    if not videos:
        sys.exit(f"[ERROR] No video files found in: {video_dir}")

    print(f"\n[Batch] Found {len(videos)} video(s) in {video_dir}")
    for i, vid in enumerate(sorted(videos), 1):
        print(f"\n{'─'*62}")
        print(f"[Batch] Processing video {i}/{len(videos)}: {vid.name}")
        result = analyse_video(
            video_path=str(vid),
            model_path=args.model,
            fps_sample=args.fps_sample,
            save_video=not args.no_video,
            movement_threshold=args.movement_threshold,
            face_threshold=args.face_threshold,
        )
        print_report(result, str(vid))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Analyse teacher enthusiasm in pre-recorded classroom video(s)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",     help="Path to a single video file")
    group.add_argument("--video_dir", help="Directory of videos to analyse (batch mode)")

    ap.add_argument("--model",              default="models/face_model.keras",
                    help="Path to trained .keras model")
    ap.add_argument("--fps_sample",         type=int,   default=30,
                    help="Analyse 1 in every N frames (default 30)")
    ap.add_argument("--no_video",           action="store_true",
                    help="Skip saving annotated output video")
    ap.add_argument("--movement_threshold", type=float, default=0.08,
                    help="Body movement score threshold (default 0.08)")
    ap.add_argument("--face_threshold",     type=float, default=0.5,
                    help="Face enthusiasm probability threshold (default 0.5)")
    args = ap.parse_args()

    if args.video:
        result = analyse_video(
            video_path=args.video,
            model_path=args.model,
            fps_sample=args.fps_sample,
            save_video=not args.no_video,
            movement_threshold=args.movement_threshold,
            face_threshold=args.face_threshold,
        )
        print_report(result, args.video)
    else:
        analyse_directory(args.video_dir, args)


if __name__ == "__main__":
    main()