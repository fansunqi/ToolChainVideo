


from visible_frames import VisibleFrames
from tools.image_captioner import ImageCaptioner
from tools.frame_selector import FrameSelector




video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
question = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."
init_video_stride = 30  # 设置抽帧间隔

# 创建可见帧管理器
visible_frames = VisibleFrames(video_path=video_path, init_video_stride=init_video_stride)

# image_captioner
image_captioner = ImageCaptioner()
image_captioner.set_frames(visible_frames)
image_captioner.inference(input="placeholder")

# 再打印信息
print("\n可见帧描述:")
print(visible_frames.get_frame_descriptions())

# frame selector
frame_selector = FrameSelector()
frame_selector.set_frames(visible_frames=visible_frames)
frame_selector.inference(input=question)


