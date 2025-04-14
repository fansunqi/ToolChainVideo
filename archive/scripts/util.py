import json
import os
import cv2
import ffmpeg

def save_to_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def parse_answer(answer):
    good_ans_list = answer.get("good_anwsers")
    if good_ans_list:
        pass
    else:
        return None

def adjust_video_resolution(video_path: str):
    # 解析视频路径
    dir_name, file_name = os.path.split(video_path)
    file_base, file_ext = os.path.splitext(file_name)
    backup_path = os.path.join(dir_name, f"{file_base}_org{file_ext}")
    
    # 获取视频信息
    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if not video_stream:
        print(f"\nError: Cannot find video stream in {video_path}")
        return
    
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    # 检查是否需要裁剪
    new_width = width if width % 2 == 0 else width - 1
    new_height = height if height % 2 == 0 else height - 1
    if new_width == width and new_height == height:
        # print("No need to crop. The resolution is already even.")
        return
    
    # 备份原视频
    os.rename(video_path, backup_path)
    
    # 处理视频
    ffmpeg.input(backup_path).filter('crop', new_width, new_height, 0, 0).output(video_path).run()
    
    print(f"\nVideo cropped to even resolution and saved as {video_path}, original saved as {backup_path}")  

if __name__ == "__main__":
    test_video_path = "/share_data/NExT-QA/NExTVideo/0071/2617504308.mp4"
    adjust_video_resolution(test_video_path)
