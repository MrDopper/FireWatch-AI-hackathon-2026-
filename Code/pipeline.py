import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from .camera import CameraError, close_display, read_samples, show_frame  # noqa: E402
from .target_box import CLIPS, smoke_target_box  # noqa: E402
from src.detector import detect as model_detect  # noqa: E402
from src.detector import loadDetector  # noqa: E402
from src.temporal import DEFAULT_CONFIG, new_temporal_state, update_temporal  # noqa: E402

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
STATE_COLOURS = {
    "Monitoring": (200, 200, 200),
    "Candidate": (0, 200, 255),
    "Review alert": RED,
    "Unavailable": (255, 120, 0),
}


def detect(engine, frame_bgr):
    """Run the Pyronear model on an OpenCV BGR frame."""

    start = time.perf_counter()
    try:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = model_detect(engine, image)
        score, boxes, valid, error = result["visual_score"], result["boxes"], True, None
    except Exception as exc:
        score, boxes, valid, error = None, [], False, f"{type(exc).__name__}: {exc}"

    return {
        "visual_score": score,
        "boxes": boxes,
        "inference_ms": (time.perf_counter() - start) * 1000,
        "valid": valid,
        "error": error,
    }


def draw_overlay(frame, target_box, result, decision, record):
    """Draw the target box (red), model boxes (green) and the evidence panel."""

    out = frame.copy()
    scale = out.shape[1] / 1280
    thick = max(2, int(3 * scale))
    font = 0.8 * scale

    if target_box:
        x1, y1, x2, y2 = target_box
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, thick)
        cv2.putText(out, "target: smoke plume", (x1 + thick * 3, y2 - thick * 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font, RED, thick)

    for x1, y1, x2, y2, conf in result["boxes"]:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, GREEN, thick)
        cv2.putText(out, f"model {conf:.3f}", (p1[0], max(int(30 * scale), p1[1] - thick * 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, GREEN, thick)

    score = record["visual_score"]
    lines = [
        (f"t = {record['source_time_s']:.2f}s  (slot {record['sample_time_s']:.1f}s)", WHITE),
        (f"Visual model score: {'n/a' if score is None else f'{score:.3f}'}"
         f"  ({record['inference_ms']:.0f} ms)", WHITE),
        (f"Immediate alert: {'YES' if record['baseline_alert'] else 'no'}", WHITE),
        (f"Temporal: {decision['state']}  "
         f"({decision['positive_samples']} of {decision['window_samples']} samples qualify)",
         STATE_COLOURS[decision["state"]]),
        (decision["reason"], STATE_COLOURS[decision["state"]]),
    ]
    line_h = int(34 * scale)
    pad = int(10 * scale)
    panel_w = int(820 * scale)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, pad * 2 + line_h * len(lines)), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(out, text, (pad, pad + line_h * (i + 1) - int(8 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font * 0.85, colour, max(1, thick - 1))
    return out


def process_clip(video_path, engine, config=DEFAULT_CONFIG, show=True, annotated_dir=None):

    clip_id = os.path.basename(video_path)
    # The footage label only decides whether a reference target box is drawn;
    # it never reaches the detector or the alert rules.
    draw_target = CLIPS.get(clip_id) == "smoke"
    state = new_temporal_state(clip_id)
    baseline_alert = False
    records = []

    if annotated_dir:
        os.makedirs(annotated_dir, exist_ok=True)

    for sample in read_samples(video_path, sample_rate=config["sample_hz"]):

        frame = sample["frame"]
        result = detect(engine, frame)
        target_box = smoke_target_box(frame) if draw_target else None

        decision = update_temporal(state, {
            "clip_id": clip_id,
            "sample_time_s": sample["sample_time_s"],
            "source_time_s": sample["source_time_s"],
            "visual_score": result["visual_score"],
            "valid": result["valid"],
            "error": result["error"],
        }, config)

        alert_event_baseline = (
            result["valid"]
            and result["visual_score"] >= config["threshold"]
            and not baseline_alert
        )
        baseline_alert = baseline_alert or alert_event_baseline

        record = {
            "clip_id": clip_id,
            "frame_index": sample["frame_index"],
            "sample_time_s": round(sample["sample_time_s"], 2),
            "source_time_s": round(sample["source_time_s"], 3),
            "visual_score": result["visual_score"],
            "boxes": json.dumps([[round(v, 1) for v in b[:4]] + [round(b[4], 4)]
                                 for b in result["boxes"]]),
            "target_box": json.dumps(target_box) if target_box else "",
            "inference_ms": round(result["inference_ms"], 1),
            "valid": result["valid"],
            "error": result["error"],
            "baseline_alert": baseline_alert,
            "alert_event_baseline": alert_event_baseline,
            "temporal_alert": decision["temporal_alert"],
            "alert_event_temporal": decision["alert_event_temporal"],
            "state": decision["state"],
            "reason": decision["reason"],
            "window_samples": decision["window_samples"],
            "positive_samples": decision["positive_samples"],
        }
        records.append(record)

        print(
            f"Time={record['source_time_s']:.2f}s "
            f"Score={result['visual_score']} "
            f"State={decision['state']} "
            f"({decision['positive_samples']}/{decision['window_samples']})"
        )

        annotated = draw_overlay(frame, target_box, result, decision, record)
        if annotated_dir:
            cv2.imwrite(
                os.path.join(annotated_dir, f"{sample['sample_time_s']:05.1f}s.jpg"),
                cv2.resize(annotated, (1280, int(annotated.shape[0] * 1280 / annotated.shape[1]))),
            )
        if show and not show_frame(annotated):
            break

    return records


def save_csv(
    records,
    output_file="results/predictions.csv"
):

    if not records:
        print("No records generated.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(
        output_file,
        "w",
        newline=""
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Saved CSV: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run FireWatch on one video clip.")
    parser.add_argument("video", nargs="?", default="data/Smoke Plume 4K Video.mp4")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, "threshold": args.threshold}
    clip_stem = Path(args.video).stem.replace(" ", "_")

    try:
        records = process_clip(
            args.video,
            loadDetector(),
            config,
            show=not args.no_display,
            annotated_dir=f"results/annotated/{clip_stem}",
        )
    except CameraError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        close_display()

    save_csv(records)

    print(
        f"Processed {len(records)} samples."
    )
