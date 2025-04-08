import cv2
import numpy as np

    
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
    