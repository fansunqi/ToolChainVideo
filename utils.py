import torchvision.io as tvio


def video_to_tensor(video_path):
    # 使用torchvision读取视频
    video, _, _ = tvio.read_video(video_path, pts_unit='sec')
    
    # 转换为张量格式 (T,H,W,C) -> (T,C,H,W)
    video = video.permute(0, 3, 1, 2)

    return video






if __name__ == "__main__":
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    video = video_to_tensor(video_path)