"""
video_tester.py
================
Give any video path — real or AI generated.
Model shows teaching mode on every frame.

Folder structure (auto-created, nothing ever overwritten):
    teachingmodes/
      test_frames/
        MyLecture_2025-06-10_14-32-05/   ← new folder per video per run
          frame_000002_boardonly.jpg
          frame_000004_boardonly.jpg  ...
      outputs/
        output_MyLecture_2025-06-10_14-32-05.mp4

Keys while playing:
  Q / ESC  → quit
  S        → save screenshot
  P        → pause / resume

CHANGE ONLY THE 2 PATHS MARKED ★ BELOW, THEN RUN:
  python video_tester.py
Or:
  python video_tester.py --video "C:\\path\\to\\video.mp4"
"""

import argparse, os, time
from datetime import datetime
from pathlib import Path
from collections import deque, Counter

import numpy as np, cv2, torch, torch.nn as nn
from torchvision import models, transforms

# ══════════════════════════════════════════════════════════════
#  ★  CHANGE PATH 1 — your trained model file
# ══════════════════════════════════════════════════════════════
DEFAULT_MODEL = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\teachingmodes\best_model.pth"
# ══════════════════════════════════════════════════════════════
#  ★  CHANGE PATH 2 — same MODEL_DIR used in model_trainer.py
# ══════════════════════════════════════════════════════════════
MODEL_DIR = r"C:\Users\LENOVO\Desktop\AISHWARYA_PJ -1MS23CS017\sem06\miniproject\teachingmodes"
# ══════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Model  (must match model_trainer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class TeachingClassifier(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        base          = models.mobilenet_v2(weights=None)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1280, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512,  128), nn.ReLU(),
            nn.Linear(128, n),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)).view(x.size(0), -1))


# ─────────────────────────────────────────────────────────────────────────────
#  Predictor
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, model_path):
        ckpt             = torch.load(model_path, map_location=DEVICE)
        self.class_names = ckpt.get("class_names", ["boardonly", "pptonly", "boardandppt"])
        img_size         = ckpt.get("img_size", 224)
        self.model       = TeachingClassifier(ckpt.get("num_classes", 3)).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # maxlen=5: reacts quickly to mode changes.
        # maxlen=20 was too large — locked onto one class for the whole video.
        self.buf = deque(maxlen=5)

    def predict(self, frame_bgr):
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tfm(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        idx    = int(np.argmax(probs))
        raw    = self.class_names[idx]
        conf   = float(probs[idx])
        self.buf.append(raw)
        smooth = Counter(self.buf).most_common(1)[0][0]
        return raw, smooth, conf, probs


# ─────────────────────────────────────────────────────────────────────────────
#  Display overlay
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "boardonly":   ( 40, 210,  60),
    "pptonly":     ( 40, 130, 255),
    "boardandppt": (  0, 160, 255),
}
LABELS = {
    "boardonly":   "BLACKBOARD ONLY",
    "pptonly":     "PPT / SCREEN ONLY",
    "boardandppt": "BOTH BOARD + PPT",
}
DESCRIPTIONS = {
    "boardonly":   ("The teacher is using only the blackboard for teaching. "
                    "Writing and explaining content directly on the board with chalk."),
    "pptonly":     ("The teacher is using only PPT / projector screen for teaching. "
                    "Presenting digital slides and explaining from the screen."),
    "boardandppt": ("The teacher is using both the blackboard and PPT screen for teaching. "
                    "Combining chalk-written content on the board with projected slides."),
}


def draw(frame, raw, smooth, conf, probs, class_names, fps, ts):
    h, w = frame.shape[:2]
    bar  = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 118), (8, 8, 8), -1)
    cv2.addWeighted(bar, 0.6, frame, 0.4, 0, frame)

    col = COLORS.get(smooth, (255, 255, 255))
    cv2.putText(frame, LABELS.get(smooth, smooth),
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

    rc = COLORS.get(raw, (160, 160, 160))
    cv2.putText(frame, f"frame: {raw}  {conf*100:.0f}%",
                (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, rc, 1, cv2.LINE_AA)

    m, s = divmod(int(ts), 60)
    cv2.putText(frame, f"{m:02d}:{s:02d}   {fps:.0f}fps",
                (w - 165, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1)

    by = 82
    for cn, p in zip(class_names, probs):
        bw = int(220 * p)
        c  = COLORS.get(cn, (110, 110, 110))
        cv2.rectangle(frame, (12, by), (12 + bw, by + 14), c, -1)
        cv2.putText(frame, f"{cn}: {p*100:.0f}%",
                    (242, by + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (200, 200, 200), 1, cv2.LINE_AA)
        by += 18

    cv2.rectangle(frame, (0, h - 48), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Teaching mode: {LABELS.get(smooth, smooth)}",
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.84, col, 2, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(seconds):
    """Float seconds → 'MM:SS' string."""
    seconds = max(0.0, float(seconds))
    m, s    = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(video_path, all_predictions, class_names,
                  output_path, frames_dir, fps_src=25.0,
                  total_frames=0, saved_images=0, true_duration=0.0):
    """
    all_predictions : list of (smooth_label, timestamp_seconds)
    total_frames    : total raw frames read (frame_n)
    saved_images    : JPG images saved to frames_dir
    true_duration   : total_frames_cap / fps  — set before loop, always exact
    """
    total = len(all_predictions)
    if total == 0:
        return

    labels    = [p[0] for p in all_predictions]
    counts    = Counter(labels)
    dominant  = counts.most_common(1)[0][0]
    frame_dur = 1.0 / max(fps_src, 1.0)

    # ── Time per mode ─────────────────────────────────────────────────────────
    # ts = frame_n / fps so consecutive gaps are always exactly frame_dur
    time_per_mode = {cls: 0.0 for cls in class_names}
    for i, (lbl, ts) in enumerate(all_predictions):
        gap = (all_predictions[i + 1][1] - ts) if i + 1 < len(all_predictions) else frame_dur
        if gap <= 0:
            gap = frame_dur
        time_per_mode[lbl] = time_per_mode.get(lbl, 0.0) + gap

    total_time = sum(time_per_mode.values())
    if total_time <= 0:
        total_time = total * frame_dur
        for cls in class_names:
            time_per_mode[cls] = counts.get(cls, 0) * frame_dur

    # ── Verdict ───────────────────────────────────────────────────────────────
    # If teacher used board independently >5% AND ppt independently >5%
    # at different points in the video → verdict = boardandppt
    pct_board = time_per_mode.get("boardonly", 0.0) / total_time * 100
    pct_ppt   = time_per_mode.get("pptonly",   0.0) / total_time * 100
    verdict   = "boardandppt" if (pct_board > 5 and pct_ppt > 5) else dominant

    # ── Print ─────────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  TEACHING MODE ANALYSIS — RESULT")
    print(sep)

    display_duration = true_duration if true_duration > 0 else total_time
    print(f"\n  Video            : {Path(video_path).name}")
    print(f"  Video duration   : {fmt_time(display_duration)}")
    print(f"  Total frames     : {total_frames}")
    print(f"  Frames analysed  : {total}")
    print(f"  Images saved     : {saved_images}")

    # Time-based breakdown — no raw frame numbers, time + % only
    print(f"\n  {'─'*54}")
    print(f"  Teaching mode breakdown  (time spent in each mode):")
    print(f"  {'─'*54}")
    for cls in class_names:
        t_cls = time_per_mode.get(cls, 0.0)
        pct_t = t_cls / total_time * 100 if total_time > 0 else 0.0
        bar   = "█" * int(pct_t / 5)
        label = LABELS.get(cls, cls)
        if t_cls > 0:
            print(f"  {label:<20s}  {fmt_time(t_cls)}  ({pct_t:5.1f}%)  {bar}")
        else:
            print(f"  {label:<20s}  {fmt_time(t_cls)}  ({pct_t:5.1f}%)")

    # Verdict — breakdown above already shows all times, no TIME IN MODE here
    print(f"\n  {'─'*54}")
    print(f"  VERDICT          : {LABELS.get(verdict, verdict)}")
    print(f"  CONCLUSION       : {DESCRIPTIONS.get(verdict, verdict)}")
    print(f"  {'─'*54}")
    print(f"\n  Annotated video  : {output_path}")
    print(f"  Saved frames at  : {frames_dir}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def get_path():
    print("\n" + "=" * 55)
    print("  Teaching Mode — Video Tester")
    print("=" * 55)
    while True:
        p = input("\n  Enter full path to video file:\n  > ").strip().strip('"').strip("'")
        if os.path.isfile(p):
            return p
        print(f"  Not found: {p}\n  Check path and try again.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip",  type=int, default=2,
                    help="Analyse 1 in every N frames (default 2)")
    args = ap.parse_args()

    video_path = args.video if args.video else get_path()

    if not os.path.isfile(video_path):
        print(f"\nERROR: Video not found:\n  {video_path}"); return
    if not os.path.isfile(args.model):
        print(f"\nERROR: Model not found:\n  {args.model}")
        print("Run model_trainer.py first."); return

    # ── Timestamped folders ───────────────────────────────────────────────────
    vid_stem    = Path(video_path).stem
    run_ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    frames_dir  = os.path.join(MODEL_DIR, "test_frames", f"{vid_stem}_{run_ts}")
    outputs_dir = os.path.join(MODEL_DIR, "outputs")
    output_path = os.path.join(outputs_dir, f"output_{vid_stem}_{run_ts}.mp4")
    os.makedirs(frames_dir,  exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"\n  Video  : {video_path}")
    print(f"  Model  : {args.model}")
    print(f"  Device : {DEVICE}")
    print(f"\n  Test frames  → {frames_dir}")
    print(f"  Output video → {output_path}\n")

    predictor = Predictor(args.model)
    print(f"  Classes : {predictor.class_names}")
    print("  Q=quit   S=screenshot   P=pause\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}"); return

    fps_src          = cap.get(cv2.CAP_PROP_FPS) or 25
    width            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    true_duration    = total_frames_cap / max(fps_src, 1.0)
    delay            = max(1, int(1000 / fps_src))

    print(f"  Video info : {width}x{height}  {fps_src:.1f}fps  "
          f"{total_frames_cap} frames  duration={fmt_time(true_duration)}\n")

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (width, height)
    )

    frame_n  = 0
    saved_n  = 0
    t_prev   = time.time()
    fps_disp = 0.0
    paused   = False
    shot_n   = 0
    display  = None

    last_raw    = last_smooth = "..."
    last_conf   = 0.0
    last_probs  = np.zeros(len(predictor.class_names))

    all_predictions = []   # each entry: (smooth_label, timestamp_seconds)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  Video finished.")
                break

            frame_n += 1
            now      = time.time()
            fps_disp = 0.88 * fps_disp + 0.12 / max(now - t_prev, 1e-6)
            t_prev   = now
            # frame_n / fps is always exact.
            # CAP_PROP_POS_MSEC after cap.read() points to the NEXT frame
            # and is unreliable on many codecs — caused the "4 sec" bug.
            ts = frame_n / max(fps_src, 1.0)

            if frame_n % args.skip == 0:
                last_raw, last_smooth, last_conf, last_probs = predictor.predict(frame)

                # Save annotated frame image
                save_frame = frame.copy()
                col = COLORS.get(last_smooth, (255, 255, 255))
                cv2.putText(save_frame,
                            f"{LABELS.get(last_smooth, last_smooth)}  {last_conf*100:.0f}%",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                cv2.putText(save_frame, f"frame {frame_n:06d}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imwrite(
                    os.path.join(frames_dir, f"frame_{frame_n:06d}_{last_smooth}.jpg"),
                    save_frame
                )
                saved_n += 1

                # Per-frame terminal line
                m_ts, s_ts = divmod(int(ts), 60)
                prob_str   = "  ".join(
                    f"{cn}:{int(p*100):3d}%"
                    for cn, p in zip(predictor.class_names, last_probs)
                )
                print(f"  [{m_ts:02d}:{s_ts:02d}] frame {frame_n:05d} "
                      f"| {LABELS.get(last_smooth, last_smooth):<20s} "
                      f"| {prob_str}")

            all_predictions.append((last_smooth, ts))

            display = draw(frame.copy(), last_raw, last_smooth,
                           last_conf, last_probs,
                           predictor.class_names, fps_disp, ts)
            writer.write(display)

        if display is not None:
            cv2.imshow("Teaching Mode Detection", display)

        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = f"screenshot_{shot_n:04d}.jpg"
            cv2.imwrite(fname, display)
            print(f"  Saved: {fname}")
            shot_n += 1
        elif key == ord("p"):
            paused = not paused
            print("  Paused." if paused else "  Resumed.")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print_summary(video_path, all_predictions,
                  predictor.class_names, output_path, frames_dir,
                  fps_src, total_frames=frame_n, saved_images=saved_n,
                  true_duration=true_duration)


if __name__ == "__main__":
    main()