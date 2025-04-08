import cv2
from ultralytics import YOLOE
from typing import List  # 用于类型注解
import pdb

# frame-level
class YOLOTrackerFrame:
    def __init__(self, model_path: str):
        self.model = YOLOE(model_path)
        print("Model loaded successfully.")

    def inference(self, 
              frames: List[cv2.Mat], 
              open_vocabulary: List[str], 
              persist: bool = True, 
              stream: bool = True, 
              save: bool = True, 
              output_id = False):
        """
        对单帧图像进行目标跟踪。
        :param frame: 输入的图像帧列表。
        :param open_vocabulary: 开放词汇表，必须是字符串列表。
        :param persist: 是否在帧之间保持跟踪。
        :return: 跟踪结果。
        """
        if not isinstance(open_vocabulary, list):
            raise TypeError(f"Expected 'open_vocabulary' to be a list, but got {type(open_vocabulary).__name__}")
        
        self.model.set_classes(open_vocabulary, self.model.get_text_pe(open_vocabulary))
        
        # TODO 检查 stream 函数
        results = self.model.track(frames, persist=persist, save=save, stream=stream)

        # result 是单独一帧的结果
        result_message = "Here are the detection results for the video clip:\n"
        for frame_idx, result in enumerate(results):
            cls = result.boxes.cls.cpu().numpy().astype(int)
            try: 
                ids = result.boxes.id.cpu().numpy().astype(int)
            except AttributeError:
                ids = [None] * len(cls)

            result_message += f"Frame {frame_idx}: "

            if output_id: 
                # TODO 完善 output_id = True 的情况
                for id, c in zip(ids, cls):
                    c_name = open_vocabulary[c]
                    result_message += f"ID {id}, Class {c_name}; "
            else:
                # 统计各类物体个数
                frame_class_counts = {cls_name: 0 for cls_name in open_vocabulary}
                for c in cls:
                    c_name = open_vocabulary[c]
                    frame_class_counts[c_name] += 1

                # 输出个数
                for cls_name, count in frame_class_counts.items():
                    if count > 0:
                        result_message += f"{count} {cls_name}"

            result_message += "\n"
        
        # TODO ReID 

        return result_message
    

if __name__ == "__main__":
    # 初始化 YOLO 模型
    open_vocabulary = ["children"]
    model_path = "checkpoints/yoloe-11l-seg.pt"
    # 视频路径
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    
    # Frame-level
    yolo_tracker = YOLOTrackerFrame(model_path)
    
    video_stride = 30  # 设置视频 stride，跳过的帧数

    from frame_selector import *
    frames = select_frames(video_path=video_path, video_stride=video_stride)
    result_message = yolo_tracker.inference(frames, 
                                            open_vocabulary=open_vocabulary)
    print(result_message)

    