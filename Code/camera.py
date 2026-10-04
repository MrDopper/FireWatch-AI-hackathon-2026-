# camera.py

import cv2
import os


def read_samples(video_path, sample_rate=2, output_folder="captured_frames"):
    """
    Read video, display it, and save sampled frames.

    Args:
        video_path: path to video
        sample_rate: frames per second to save
        output_folder: folder for saved images
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise Exception("Invalid FPS")

    step = max(1, round(fps / sample_rate))

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Show video
        cv2.imshow("FireWatch Camera Feed", frame)

        source_time_s = frame_index / fps

        # Save sampled frames
        if frame_index % step == 0:

            filename = (
                f"frame_{saved_count:04d}_"
                f"{source_time_s:.1f}s.jpg"
            )

            save_path = os.path.join(output_folder, filename)

            cv2.imwrite(save_path, frame)

            print(
                f"Saved {filename} "
                f"(time={source_time_s:.2f}s)"
            )

            saved_count += 1

        frame_index += 1

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nFinished.")
    print(f"Frames saved: {saved_count}")
    print(f"Folder: {output_folder}")


if __name__ == "__main__":
    read_samples(
        "Photos/Dog.mp4",
        sample_rate=2,
        output_folder="captured_frames"
    )