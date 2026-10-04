# camera.py

import os

import cv2


class CameraError(Exception):
    pass


WINDOW_NAME = "FireWatch Camera Feed"


def read_samples(
    video_path,
    sample_rate=2,
    output_folder="captured_frames"
):
    """
    Read a video, save sampled frames, and yield frame information.

    sample_rate=2 means one sample per scheduled half-second slot of source
    time (0.0s, 0.5s, 1.0s, ...). Each slot takes the decoded frame closest to
    it, so the schedule does not drift on fractional FPS such as 23.976.
    """

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise CameraError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise CameraError("Invalid FPS")

    period = 1 / sample_rate
    half_frame = 0.5 / fps

    frame_index = 0
    slot_index = 0
    saved_count = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            source_time_s = frame_index / fps
            sample_time_s = slot_index * period

            if source_time_s + half_frame >= sample_time_s:

                if output_folder:
                    filename = (
                        f"frame_{saved_count:04d}_"
                        f"{sample_time_s:.1f}s.jpg"
                    )
                    cv2.imwrite(
                        os.path.join(output_folder, filename),
                        frame
                    )

                yield {
                    "frame": frame,
                    "frame_index": frame_index,
                    "sample_time_s": sample_time_s,
                    "source_time_s": source_time_s,
                }

                saved_count += 1
                slot_index += 1

            frame_index += 1

    finally:

        cap.release()

        print("\nFinished.")
        print(f"Frames saved: {saved_count}")
        if output_folder:
            print(f"Folder: {output_folder}")


def show_frame(frame, max_width=1280, delay_ms=1):
    """Display a frame scaled to max_width. Returns False if q was pressed."""

    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(
            frame,
            (max_width, int(height * max_width / width))
        )

    cv2.imshow(WINDOW_NAME, frame)

    return (cv2.waitKey(delay_ms) & 0xFF) != ord("q")


def close_display():
    cv2.destroyAllWindows()


if __name__ == "__main__":

    VIDEO_PATH = "data/*.mp4"

    try:

        for sample in read_samples(VIDEO_PATH, sample_rate=2):
            print(
                f"Frame {sample['frame_index']} | "
                f"Slot {sample['sample_time_s']:.1f}s | "
                f"Time {sample['source_time_s']:.2f}s"
            )
            if not show_frame(sample["frame"], delay_ms=500):
                break

    except CameraError as e:
        print(f"Error: {e}")

    finally:
        close_display()
