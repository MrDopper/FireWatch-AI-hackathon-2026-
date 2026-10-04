import csv
import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera import read_samples
from src.detector import detect as detect_image
from src.detector import loadDetector


def detect(engine, frame):
    """Run the detector on an OpenCV BGR frame."""

    start = time.perf_counter()

    try:
        image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = detect_image(engine, image)

        return {
            "visual_score": result["visual_score"],
            "boxes": result["boxes"],
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
            "valid": True,
            "error": None,
        }

    except Exception as e:
        return {
            "visual_score": None,
            "boxes": [],
            "inference_ms": round((time.perf_counter() - start) * 1000, 1),
            "valid": False,
            "error": str(e),
        }


def process_clip(video_path):

    records = []

    engine = loadDetector()

    for sample in read_samples(video_path):

        result = detect(engine, sample["frame"])

        record = {
            "frame_index": sample["frame_index"],
            "source_time_s": round(
                sample["source_time_s"],
                2
            ),
            "visual_score": result["visual_score"],
            "inference_ms": result["inference_ms"],
            "valid": result["valid"],
            "error": result["error"],
        }

        records.append(record)

        print(
            f"Time={record['source_time_s']}s "
            f"Score={record['visual_score']}"
        )

    return records


def save_csv(
    records,
    output_file="results/predictions.csv"
):

    if not records:
        print("No records generated.")
        return

    os.makedirs("results", exist_ok=True)

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

    video_path = "Photos/flame.mp4"

    records = process_clip(video_path)

    save_csv(records)

    print(
        f"Processed {len(records)} samples."
    )