import cv2
import numpy as np
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class Frame:
    """表示视频中的一帧"""
    index: int  # 帧在视频中的索引
    timestamp: float  # 帧的时间戳（秒）
    image: np.ndarray  # 帧的图像数据
    description: Optional[str] = None  # 帧的文字描述

class VisibleFrames:
    """管理一个视频中的可见帧"""
    def __init__(self):
        self.frames: List[Frame] = []  # 存储所有可见帧
        self.video_info: Optional[Dict] = None  # 存储视频信息
    
    def add_frames(self, frames: List[np.ndarray], frame_indices: List[int], video_info: Dict):
        """添加新的可见帧"""
        self.video_info = video_info
        for frame, idx in zip(frames, frame_indices):
            timestamp = idx / video_info['fps']
            self.frames.append(Frame(
                index=idx,
                timestamp=timestamp,
                image=frame
            ))
        # 按时间戳排序
        self.frames.sort(key=lambda x: x.timestamp)
    
    def get_frame_descriptions(self) -> str:
        """获取所有可见帧的文字描述"""
        if not self.frames:
            return "No visible frames."
        
        descriptions = []
        for frame in self.frames:
            time_str = str(timedelta(seconds=int(frame.timestamp)))
            desc = f"Frame {frame.index}"
            if frame.description:
                desc += f": {frame.description}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def get_frame_count(self) -> int:
        """获取可见帧数量"""
        return len(self.frames)
    
    def get_time_range(self) -> tuple:
        """获取可见帧的时间范围"""
        if not self.frames:
            return (0, 0)
        return (self.frames[0].timestamp, self.frames[-1].timestamp)
    
    def get_frame_at_time(self, timestamp: float) -> Optional[Frame]:
        """获取指定时间点最接近的帧"""
        if not self.frames:
            return None
        return min(self.frames, key=lambda x: abs(x.timestamp - timestamp))
    
    def get_frame_indices(self) -> tuple:
        """获取可见帧的索引范围
        
        返回:
            tuple: (起始帧索引, 结束帧索引)
        """
        if not self.frames:
            return (0, 0)
        return (self.frames[0].index, self.frames[-1].index)

def get_video_info(video_path):
    """
    获取视频的基本信息，包括总帧数、帧率和时长
    
    参数:
        video_path: 视频文件路径
    
    返回:
        dict: 包含视频信息的字典
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # 释放视频捕获对象
    cap.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration  # 单位：秒
    }

def select_frames(video_path, 
                frame_indices=None,
                video_stride=None):
    """
    从视频中读取多个特定帧或根据stride进行抽帧
    
    参数:
        video_path: 视频文件路径
        frame_indices: 要读取的帧索引列表，如果为None则根据video_stride抽帧
        video_stride: 抽帧间隔，当frame_indices为None时使用
    
    返回:
        list: 读取的帧列表
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定要读取的帧索引
    if frame_indices is None:
        if video_stride is None or video_stride <= 0:
            # 如果没有指定stride或stride无效，则读取所有帧
            frame_indices = list(range(total_frames))
        else:
            # 根据stride抽帧
            frame_indices = list(range(0, total_frames, video_stride))
    
    frames = []
    
    for frame_idx in frame_indices:
        # 检查帧索引是否有效
        if frame_idx < 0 or frame_idx >= total_frames:
            print(f"Warning: Frame index {frame_idx} is out of range [0, {total_frames-1}]")
            continue
        
        # 设置要读取的帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取指定帧
        ret, frame = cap.read()
        
        # 检查是否成功读取帧
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Could not read frame {frame_idx}")
    
    # 释放视频捕获对象
    cap.release()
    
    return frames


class FrameSelector:
    def __init__(self, conf):
        # llm
        self.llm = ChatOpenAI(
            api_key = conf.openai.GPT_API_KEY,
            model = conf.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = conf.openai.PROXY
        )
        self.visible_frames = VisibleFrames()
    
    def inference(self, input):
        # input: 可见信息
        # output: 是否增加可见帧
        pass


if __name__ == "__main__":
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    video_stride = 30  # 设置抽帧间隔
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    if video_info:
        print(f"视频总帧数: {video_info['total_frames']}")
        print(f"视频帧率: {video_info['fps']:.2f} fps")
        print(f"视频时长: {video_info['duration']:.2f} 秒")
    
    # 创建可见帧管理器
    visible_frames = VisibleFrames()
    
    # 抽帧并添加到可见帧管理器
    frames = select_frames(video_path=video_path, video_stride=video_stride)
    frame_indices = list(range(0, video_info['total_frames'], video_stride))
    visible_frames.add_frames(frames, frame_indices, video_info)
    
    # 打印可见帧信息
    print("\n可见帧信息:")
    print(f"可见帧数量: {visible_frames.get_frame_count()}")
    start_idx, end_idx = visible_frames.get_frame_indices()
    print(f"帧索引范围: {start_idx} - {end_idx}")
    print("\n可见帧描述:")
    print(visible_frames.get_frame_descriptions())


    