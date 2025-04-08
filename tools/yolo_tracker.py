import cv2
from ultralytics import YOLOE
from typing import List  # 用于类型注解
import pdb
import torchvision.io as tvio
# from utils import *
import numpy as np

def video_to_numpy(video_path, max_frames=None):
    """
    将视频读取为NumPy数组
    
    参数:
        video_path: 视频文件路径
        max_frames: 最大帧数，如果为None则处理所有帧
    
    返回:
        numpy.ndarray: 形状为 [frames, height, width, channels] 的数组
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # 读取第一帧获取尺寸
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    height, width = first_frame.shape[:2]
    
    # 初始化数组
    video_array = np.zeros((total_frames, height, width, 3), dtype=np.uint8)
    
    # 重置视频捕获
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 读取所有帧
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 存储到视频数组中
        video_array[i] = frame
    
    cap.release()
    return video_array

def video_to_tensor(video_path):
    # 使用torchvision读取视频
    video, _, _ = tvio.read_video(video_path, pts_unit='sec')
    
    # 转换为张量格式 (T,H,W,C) -> (T,C,H,W)
    video = video.permute(0, 3, 1, 2)

    return video

# frame-level
class YOLOTrackerFrame:
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
    

class YOLOTrackerVideo:
    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        print("Model loaded successfully.")
    
    def inference(self, 
                video, 
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
        
        results = self.model.track(video,
                                show=False,
                                save=False,
                                stream=True)

        # result 是单独一帧的结果
        result_message = "Here are the detection results for the video clip:\n"
        for frame_idx, result in enumerate(results):
            cls = result.boxes.cls.cpu().numpy().astype(int)
            try: 
                ids = result.boxes.id.cpu().numpy().astype(int)
            except AttributeError:
                ids = [None] * len(cls)
            print(cls)
            print(ids)

            result_message += f"Frame {frame_idx}: "
            for id, c in zip(ids, cls):
                c_name = open_vocabulary[c]
                result_message += f"ID {id}, Class {c_name}; "
            result_message += "\n"
        
        return result_message
    

if __name__ == "__main__":
    # 初始化 YOLO 模型
    open_vocabulary = ["children"]
    model_path = "checkpoints/yoloe-11l-seg.pt"
    

    # 视频路径
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    
    # Frame-level
    cap = cv2.VideoCapture(video_path)
    yolo_tracker = YOLOTrackerFrame(model_path)
    
    frame_count = 0  # 初始化帧计数器
    video_stride = 30  # 设置视频 stride，跳过的帧数

    # 遍历视频帧
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1  # 更新帧计数器

            # 跳过不需要处理的帧
            if frame_count % video_stride != 0:
                continue

            print(f"Processing frame {frame_count}...")  # 打印当前帧数

            result_message = yolo_tracker.inference(frame, 
                                                    open_vocabulary=open_vocabulary, 
                                                    persist=True)
            
            print(result_message)
            # break
                
        else:
            break
    
    # 释放视频资源
    cap.release()
    
    
    # Video-level
    # video = video_to_numpy(video_path)
    # yolo_tracker = YOLOTrackerVideo(video)
    
    # result_message = yolo_tracker.inference(video_path, 
    #                                 open_vocabulary=open_vocabulary)

    # print(result_message)

    