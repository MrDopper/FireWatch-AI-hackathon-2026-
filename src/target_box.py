"""Draw a target box around the smoke plume in sampled frames of each clip.

Smoke is bright and grey/white (low colour saturation), which separates it from
blue sky and dark forest. The largest such region becomes the target box.
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
    "smoke.mp4": "smoke",
    "wildfire.mp4": "smoke",
    "fog.mp4": "fog",
}

SAMPLE_RATE = 2  # frames per source second
MAX_SATURATION = 70
MIN_VALUE = 110
MIN_AREA_FRACTION = 0.01


def smoke_target_box(frame):
    """Return [x1, y1, x2, y2] in frame pixels around the smoke plume, or None."""
    small = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < MAX_SATURATION) & (hsv[..., 2] > MIN_VALUE)).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = MIN_AREA_FRACTION * mask.size
    regions = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= min_area]
    if not regions:
        return None

    x1 = min(s[cv2.CC_STAT_LEFT] for s in regions)
    y1 = min(s[cv2.CC_STAT_TOP] for s in regions)
    x2 = max(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in regions)
    y2 = max(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in regions)
    scale = frame.shape[1] / small.shape[1]
    return [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)]


def process(name, label, writer):
    cap = cv2.VideoCapture(str(DATA / name))
    if not cap.isOpened():
        raise FileNotFoundError(DATA / name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / SAMPLE_RATE))
    out_dir = OUT / Path(name).stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            t = index / fps
            box = smoke_target_box(frame) if label == "smoke" else None
            if box:
                cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255), 8)
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
