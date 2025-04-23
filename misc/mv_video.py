import os

def find_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

video_path_org = "/share_data/NExT-QA/NExTVideo/"

all_video_filepath = find_mp4_files(video_path_org)