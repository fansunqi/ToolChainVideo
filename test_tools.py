


from tools.frame_selector import VisibleFramesInfo, select_frames, get_video_info
from tools.image_captioner import ImageCaptioner


image_captioner = ImageCaptioner()


# 在 set frames 的同时，也需要加入并维护 visible_frames_info 这个数据结构



video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
video_info = get_video_info(video_path)

# 创建可见帧管理器
visible_frames_info = VisibleFramesInfo(video_info=video_info)
    
# 抽帧并添加到可见帧管理器
video_stride = 30  # 设置抽帧间隔
frames = select_frames(video_path=video_path, video_stride=video_stride)

frame_indices = list(range(0, video_info['total_frames'], video_stride))
visible_frames_info.add_frames(frames, frame_indices)
print(visible_frames_info.get_frame_descriptions())