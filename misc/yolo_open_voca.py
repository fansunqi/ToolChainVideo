from ultralytics import YOLOE
# Initialize a YOLOE model

model_path = "checkpoints/yoloe-11l-seg-pf.pt"
model = YOLOE(model_path)
print("Model loaded successfully.")


video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"

# 直接预测 video
resuts = model.predict(video_path,
                       show=False,
                       save=True,
                       show_conf=False,
                       stream=True)

for r in resuts:
    boxes = r.boxes.xyxy.cpu().tolist()
    cls = r.boxes.cls.cpu().tolist()
    for b, c in zip(boxes, cls):
        print(f"box: {b}, cls: {model.names[c]}")
    
    # masks = r.masks  # Masks object for segment masks outputs
    # probs = r.probs  # Class probabilities for classification outputs
