import cv2
import os


class CameraError(Exception):
    pass


def read_samples(
    video_path,
    sample_rate=2,
    output_folder="captured_frames"
):
    """
    Reads a video and yields sampled frames.

    sample_rate=2
    means 2 frames per second.
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise CameraError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise CameraError("Invalid FPS")

    step = max(1, round(fps / sample_rate))

    frame_index = 0
    saved_count = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            source_time_s = frame_index / fps

            if frame_index % step == 0:

                filename = (
                    f"frame_{saved_count:04d}_"
                    f"{source_time_s:.1f}s.jpg"
                )

                save_path = os.path.join(
                    output_folder,
                    filename
                )

                cv2.imwrite(save_path, frame)

                print(
                    f"Saved {filename} "
                    f"(time={source_time_s:.2f}s)"
                )

                yield {
                    "frame": frame,
                    "frame_index": frame_index,
                    "source_time_s": source_time_s,
                }

                saved_count += 1

            frame_index += 1

    finally:
        cap.release()


if __name__ == "__main__":

    for sample in read_samples("Photos/Dog.mp4"):
        print(
            sample["frame_index"],
            sample["source_time_s"]
        )