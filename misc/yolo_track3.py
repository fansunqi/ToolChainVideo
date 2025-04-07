import cv2
from ultralytics import YOLOE
import pdb
# Initialize a YOLOE model

model_path = "../checkpoints/yoloe-11l-seg.pt"
model = YOLOE(model_path)
print("Model loaded successfully.")

names = ["children"]
model.set_classes(names, model.get_text_pe(names))

video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0  # 初始化帧计数器

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_count += 1  # 更新帧计数器
        print(f"Processing frame {frame_count}...")  # 打印当前帧数
        
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        
        assert len(results) == 1, "YOLO11 should only return one result for each frame."
        r = results[0]
        
        boxes = r.boxes.xyxy.cpu().tolist()
        cls = r.boxes.cls.cpu().tolist()
        
        # try:
        #     ids = r.boxes.id.cpu().tolist()
        # except AttributeError:
        #     pdb.set_trace()
        #     ids = [None] * len(boxes)
            
        # for b, c, id in zip(boxes, cls, ids):
        #     print(f"box: {b}, cls: {model.names[c]}, id: {id}")
        
        for b, c in zip(boxes, cls):
            print(f"box: {b}, cls: {model.names[c]}")

        # Display the annotated frame
        # cv2.imshow("YOLO11 Tracking", annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()