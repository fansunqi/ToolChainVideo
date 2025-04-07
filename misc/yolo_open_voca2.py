from ultralytics import YOLOE
import pdb
# Initialize a YOLOE model

model_path = "../checkpoints/yoloe-11l-seg.pt"
model = YOLOE(model_path)
print("Model loaded successfully.")

names = ["children"]
model.set_classes(names, model.get_text_pe(names))

video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
results = model.predict(video_path,
                       show=False,
                       save=True,
                       show_conf=False,
                       stream=True)

for r in results:
    print(r.names)
    # pdb.set_trace()
    # boxes = r.boxes.xyxy.cpu().tolist()
    # cls = r.boxes.cls.cpu().tolist()
    # for b, c in zip(boxes, cls):
    #     print(f"box: {b}, cls: {model.names[c]}")

    # masks = r.masks  # Masks object for segment masks outputs
    # probs = r.probs  # Class probabilities for classification outputs
