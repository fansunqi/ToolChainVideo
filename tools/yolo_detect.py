import cv2
from ultralytics import YOLOE
from typing import List  # 用于类型注解
import pdb

# frame-level
class YOLODetectorFrame:
    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        print("Model loaded successfully.")

    def inference(self, 
              frame: cv2.Mat, 
              open_vocabulary: List[str], 
              persist: bool = True):
        """
        对单帧图像进行目标跟踪。
        :param frame: 输入的图像帧。
        :param open_vocabulary: 开放词汇表，必须是字符串列表。
        :param persist: 是否在帧之间保持跟踪。
        :return: 跟踪结果。
        """
        if not isinstance(open_vocabulary, list):
            raise TypeError(f"Expected 'open_vocabulary' to be a list, but got {type(open_vocabulary).__name__}")
        
        self.model.set_classes(open_vocabulary, self.model.get_text_pe(open_vocabulary))
        
        results = self.model.track(frame, persist=persist)
        assert len(results) == 1, "YOLO11 should only return one result for each frame."
        
        result = results[0]
        result_message = "Here are the detection results:"
        boxes = result.boxes.xyxy.cpu().tolist()
        cls = result.boxes.cls.cpu().tolist()
        for b, c in zip(boxes, cls):
            result_message += f"\nbox: {b}, cls: {self.model.names[c]}"
        
        return result_message
    

class YOLODetectorVideo:
    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        print("Model loaded successfully.")
    
    def inference(self, 
                video_path: str, 
                open_vocabulary: List[str]):
        """
        对视频进行目标跟踪。
        :param frame: 输入的视频。
        :param open_vocabulary: 开放词汇表，必须是字符串列表。
        :return: 跟踪结果。
        """
        if not isinstance(open_vocabulary, list):
            raise TypeError(f"Expected 'open_vocabulary' to be a list, but got {type(open_vocabulary).__name__}")
        
        self.model.set_classes(open_vocabulary, self.model.get_text_pe(open_vocabulary))
        
        results = self.model.track(video_path,
                                show=False,
                                save=False)

        for result in results:
            try:
                ids = result.boxes.id.cpu().numpy().astype(int)
            except AttributeError:
                ids = None
            print(ids)
    

if __name__ == "__main__":
    # 初始化 YOLO 模型
    open_vocabulary = ["children"]
    model_path = "checkpoints/yoloe-11l-seg.pt"
    

    # 视频路径
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    cap = cv2.VideoCapture(video_path)

    
    # Frame-level
    # yolo_detector = YOLODetectorFrame(model_path)
    
    # frame_count = 0  # 初始化帧计数器
    # video_stride = 30  # 设置视频 stride，跳过的帧数

    # # 遍历视频帧
    # while cap.isOpened():
    #     success, frame = cap.read()

    #     if success:
    #         frame_count += 1  # 更新帧计数器

    #         # 跳过不需要处理的帧
    #         if frame_count % video_stride != 0:
    #             continue

    #         print(f"Processing frame {frame_count}...")  # 打印当前帧数

    #         result_message = yolo_detector.inference(frame, 
    #                                                 open_vocabulary=open_vocabulary, 
    #                                                 persist=True)
            
    #         print(result_message)
    #         break
                
    #     else:
    #         break
    
    # 释放视频资源
    cap.release()
    
    
    # Video-level
    yolo_detector = YOLODetectorVideo(model_path)
    
    results = yolo_detector.inference(video_path, 
                                      open_vocabulary=open_vocabulary)

    