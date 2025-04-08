import cv2
from ultralytics import YOLOE
from typing import List  # 用于类型注解
import pdb

def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

# frame-level
class YOLOTracker:
    def __init__(self, 
                model_path: str,
                persist: bool = True, 
                stream: bool = True, 
                save: bool = True, 
                output_id: bool = False):
        
        self.model = YOLOE(model_path)
        print("Model loaded successfully.")

        self.frames = None

        self.persist = persist
        self.stream = stream
        self.save = save
        self.output_id = output_id

    def set_frames(self, frames: List[cv2.Mat]):
        self.frames = frames

    def track_frames(self, 
              open_vocabulary: List[str]):
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
        results = self.model.track(self.frames, 
                                persist=self.persist, 
                                save=self.save, 
                                stream=self.stream)

        # result 是单独一帧的结果
        result_message = "Here are the detection and tracking results for the video clip:\n"
        for frame_idx, result in enumerate(results):
            cls = result.boxes.cls.cpu().numpy().astype(int)
            try: 
                ids = result.boxes.id.cpu().numpy().astype(int)
            except AttributeError:
                ids = [None] * len(cls)

            result_message += f"Frame {frame_idx}: "

            if self.output_id: 
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
    

    @prompts(
        name = "object-tracking-tool",
        description = "Useful when you need to detect and track objects in the video."
        "The input to this tool must be an object to be tracked, for example, children, dog, apple.",
    )
    def inference(self, input: str):
        objects_to_track = [input]
        result_message = self.track_frames(objects_to_track)
        return result_message

    

if __name__ == "__main__":
    # 初始化 YOLO 模型
    open_vocabulary = ["children"]
    model_path = "checkpoints/yoloe-11l-seg.pt"
    
    # e.g.1
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    question_w_options = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
    object_to_track = "children"

    # Frame-level
    yolo_tracker = YOLOTracker(model_path=model_path)
    
    video_stride = 30  # 设置视频 stride，跳过的帧数

    from frame_selector import *
    frames = select_frames(video_path=video_path, video_stride=video_stride)
    yolo_tracker.set_frames(frames)

    result_message = yolo_tracker.inference(object_to_track)
    print(result_message)

    