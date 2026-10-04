#capture a picture the video from every second 

import cv2 as cv
import os

capture = cv.VideoCapture('Photos/Dog.mp4')

# Create output folder
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

fps = capture.get(cv.CAP_PROP_FPS)
frame_interval = int(fps)  # frames per second

frame_count = 0
saved_count = 0

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    # Save one frame every second
    if frame_count % frame_interval == 0:
        filename = os.path.join(
            output_folder,
            f"frame_{saved_count}.jpg"
        )
        cv.imwrite(filename, frame)
        print(f"Saved {filename}")
        saved_count += 1

    cv.imshow('Dog', frame)

    frame_count += 1

    if cv.waitKey(25) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()