"""Draw a target box around the smoke plume in sampled frames of each clip.

Smoke is bright, grey/white (low saturation, not blue) and it moves/changes
between frames. The target box is built from a colour mask AND a motion mask,
the single largest region is kept, and the box is smoothed over time.
Fog clips are labelled negative, so they get no box.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "targets"

CLIPS = {
    "Smoke Plume 4K Video.mp4": "smoke",
    "Wildfire Video 4k.mp4": "smoke",
    "Fog Over Forest 4k Video.mp4": "fog",
    "smoke.mp4": "smoke",
}

SAMPLE_RATE = 2  # frames per source second

# --- detection parameters (tune for your footage) ---
MAX_SATURATION = 110       # smoke is almost colourless
MIN_VALUE = 90          # and fairly bright
MAX_BLUE_EXCESS = 15      # (B - R) above this -> sky, not smoke
MOTION_DIFF = 6           # difference vs background (0..255)
BG_ALPHA = 0.1            # background update speed
MIN_AREA_FRACTION = 0.01
WARMUP_SAMPLES = 3        # frames used to build the background before motion mask is used


class SmokeTargetTracker:
    """Stateful target box: colour + motion mask, one best region, EMA smoothing.

    Create one tracker per clip and call it on every sampled frame.
    """

    def __init__(self, width=960, ema=0.5, hold=2):
        self.width = width
        self.ema = ema          # weight of the new box when smoothing
        self.hold = hold        # frames to keep the old box if smoke disappears
        self.bg = None
        self.n = 0
        self.box = None
        self.missed = 0

    def _masks(self, small):
        blur = cv2.GaussianBlur(small, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        b = blur[..., 0].astype(np.int16)
        r = blur[..., 2].astype(np.int16)
        color = (
            (hsv[..., 1] < MAX_SATURATION)
            & (hsv[..., 2] > MIN_VALUE)
            & ((b - r) < MAX_BLUE_EXCESS)      # cut off blue sky
        ).astype(np.uint8)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.bg is None:
            self.bg = gray.copy()
        diff = cv2.absdiff(gray, self.bg)
        cv2.accumulateWeighted(gray, self.bg, BG_ALPHA)
        motion = (diff > MOTION_DIFF).astype(np.uint8)
        motion = cv2.dilate(motion, np.ones((15, 15), np.uint8))
        return color, motion

    @staticmethod
    def _best_region(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        if stats[best, cv2.CC_STAT_AREA] < MIN_AREA_FRACTION * mask.size:
            return None
        ys, xs = np.where(labels == best)
        # percentiles instead of min/max: outliers don't stretch the box
        x1, x2 = np.percentile(xs, [2, 98])
        y1, y2 = np.percentile(ys, [2, 98])
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def __call__(self, frame):
        """Return [x1, y1, x2, y2] in frame pixels, or None."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(self.width * h / w)))
        color, motion = self._masks(small)

        found = None
        if self.n >= WARMUP_SAMPLES:
            found = self._best_region(color & motion)
        if found is None:
            found = self._best_region(color)   # fallback: colour only
        self.n += 1

        if found is None:
            self.missed += 1
            if self.missed > self.hold:
                self.box = None
        else:
            self.missed = 0
            self.box = found if self.box is None else (
                self.ema * found + (1 - self.ema) * self.box
            )

        if self.box is None:
            return None
        scale = w / small.shape[1]
        return [int(v * scale) for v in self.box]


def smoke_target_box(frame):
    """Stateless single call (no motion mask). Prefer SmokeTargetTracker."""
    return SmokeTargetTracker()(frame)


def process(name, label, writer):
    cap = cv2.VideoCapture(str(DATA / name))
    if not cap.isOpened():
        raise FileNotFoundError(DATA / name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / SAMPLE_RATE))
    out_dir = OUT / Path(name).stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = SmokeTargetTracker() if label == "smoke" else None

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            t = index / fps
            box = tracker(frame) if tracker else None
            if box:
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 8)
                cv2.putText(frame, "target: smoke", (box[0] + 10, box[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.imwrite(str(out_dir / f"{t:05.1f}s.jpg"), cv2.resize(frame, (1280, 720)))
            writer.writerow([name, label, index, round(t, 2), *(box or ["", "", "", ""])])
        index += 1
    cap.release()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "targets.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "label", "frame_index", "source_time_s", "x1", "y1", "x2", "y2"])
        for name, label in CLIPS.items():
            process(name, label, writer)
            print("done", name)
    print("Saved", OUT / "targets.csv")