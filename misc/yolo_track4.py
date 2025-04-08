import cv2
from ultralytics import YOLOE
import pdb
import numpy as np
import os

from collections import defaultdict

# Initialize a YOLOE model

model_path = "../checkpoints/yoloe-11l-seg.pt"
model = YOLOE(model_path)
print("Model loaded successfully.")

names = ["children"]
model.set_classes(names, model.get_text_pe(names))

video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

frame_count = 0  # 初始化帧计数器
video_stride = 2  # 设置视频 stride，跳过的帧数

# 创建保存图片的目录
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1  # 更新帧计数器
        
        # 跳过不需要处理的帧
        if frame_count % video_stride != 0:
            continue
        
        print(f"Processing frame {frame_count}...")  # 打印当前帧数
        
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        assert len(results) == 1, "YOLO11 should only return one result for each frame."
        r = results[0]
        
        boxes = r.boxes.xyxy.cpu().tolist()
        cls = r.boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        annotated_frame = results[0].plot()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        
        # 保存当前帧为图片
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved frame {frame_count} to {output_path}")
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()