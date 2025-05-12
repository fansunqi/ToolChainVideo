import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频的帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔指定帧数保存一帧
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"完成！共保存了 {saved_count} 帧到目录: {output_dir}")

# 示例用法
video_path = "beast.webm"  # 替换为你的 .webm 文件路径
output_dir = "beast_extracted_frames"  # 替换为你想保存图片的目录
extract_frames(video_path, output_dir, fps=1)