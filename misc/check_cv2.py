import cv2
path = '/share_data/NExT-QA/NExTVideo/0071/2617504308.mp4'
cap = cv2.VideoCapture(path)


vid_stride = 30
for _ in range(vid_stride):
    cap.grab()

success, im0 = cap.retrieve()
print("done!")