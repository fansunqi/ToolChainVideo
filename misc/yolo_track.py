from ultralytics import YOLO
import pdb

# Load an official or custom model
model_path = "../checkpoints/yolo11n.pt"
model = YOLO(model_path)  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack

video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"

# 其它的参数: conf, iou
# tracker_selection: tracker 所采用的方法
# persist: 一阵一阵处理时，是否告诉模型这是下一帧
results = model.track(video_path,
                    show=False,
                    save=True)


for r in results:
    print(r.names)
