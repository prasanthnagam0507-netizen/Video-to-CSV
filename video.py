import cv2
import os

# Biology lecture video
video_path = r"videos\biology.mp4"

# Folder for extracted frames
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video:", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = frame_count / fps

print("Video opened successfully!")
print("FPS:", fps)
print("Frame count:", frame_count)
print("Resolution:", width, "x", height)
print("Duration:", round(duration, 2), "seconds")

# Extract one frame every 5 seconds
interval = 5
frame_interval = int(fps * interval)

frame_number = 0
saved = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:

        filename = os.path.join(
            output_folder,
            f"frame_{saved:04d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved += 1

    frame_number += 1

cap.release()

print("Frames extracted:", saved)