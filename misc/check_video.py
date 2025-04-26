import cv2

video_path = "/mnt/Shared_05_disk/fsq/VideoMME/data/Bkheu99K5lY.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Width: {width}, Height: {height}")
cap.release()