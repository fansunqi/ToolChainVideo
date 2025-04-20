import json
import os
import cv2
import ffmpeg
import shutil
import pickle
from langchain_core.tools import Tool

def save_to_json(output_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

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


def backup_file(opt, conf, timestamp):
    # 将 main.py 文件自身和 opt.config 文件复制一份存储至 conf.output_path
    current_script_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    shutil.copy(current_script_path, os.path.join(conf.output_path, f"main_{timestamp}.py"))
    # 复制 opt.config 文件到输出目录
    config_basename = os.path.basename(opt.config).split('.')[0]
    shutil.copy(opt.config, os.path.join(conf.output_path, f"{config_basename}_{timestamp}.yaml"))   
    

def load_cache(mannual_cache_file):
    if os.path.exists(mannual_cache_file):
        print(f"Loading LLM cache from {mannual_cache_file}...\n")
        with open(mannual_cache_file, "rb") as f:
            mannual_cache = pickle.load(f)
    else:
        print(f"Creating LLM cache: {mannual_cache_file}...\n")
        mannual_cache = {}
    return mannual_cache


def save_cache(mannual_cache, query, steps, mannual_cache_file):
    mannual_cache[query] = steps
    print("\nSaving cache...")
    with open(mannual_cache_file, "wb") as f:
        pickle.dump(mannual_cache, f)


def get_tools(conf):
    tool_list = conf.tool.tool_list
    tool_instances = []
    for tool_name in tool_list:
        tool_instances.append(globals()[tool_name](conf))
    print(f"tool_instances: {str(tool_instances)}")
    
    tools = []
    for tool_instance in tool_instances:
        for e in dir(tool_instance):
            if e.startswith("inference"):
                func = getattr(tool_instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func))
    print(f"tools: {str(tools)}")
    
    return tool_instances, tools


if __name__ == "__main__":
    test_video_path = "/share_data/NExT-QA/NExTVideo/0071/2617504308.mp4"
    adjust_video_resolution(test_video_path)
