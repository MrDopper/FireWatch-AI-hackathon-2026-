import csv
import os

from camera import read_samples


# Temporary fake detector
# Replace later with:
# from detector import detect

def detect(frame):

    return {
        "visual_score": 0.75,
        "boxes": [],
        "inference_ms": 12.5,
        "valid": True,
        "error": None,
    }


def process_clip(video_path):

    records = []

    for sample in read_samples(video_path):

        result = detect(sample["frame"])

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

    video_path = "Photos/Dog.mp4"

    records = process_clip(video_path)

    save_csv(records)

    print(
        f"Processed {len(records)} samples."
    )