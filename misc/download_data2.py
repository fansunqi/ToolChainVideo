# from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import string
import os
import argparse

# Login to Hugging Face
login(token="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")

def download_all_parts(local_dir="downloads"):
    # Create the download directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Create list of all suffixes: ab, ac, ..., az, ba, bb, bc, bd, be
    suffixes = []
    # First add ab through az (skipping aa)
    suffixes.extend([f"a{c}" for c in string.ascii_lowercase])
    # Then add ba through be
    suffixes.extend([f"b{c}" for c in string.ascii_lowercase[:5]])  # only up to 'be'
    
    # Download each part
    for suffix in suffixes:
        filename = f"videos.tar.part.{suffix}"
        print(f"Downloading {filename}...")
        try:
            file_path = hf_hub_download(
                repo_id="longvideobench/LongVideoBench",
                repo_type="dataset",
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False  # 直接下载文件而不是创建符号链接
            )
            print(f"Downloaded {filename} to: {file_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download video parts from Hugging Face')
    parser.add_argument('--download_dir', type=str, default='/mnt/Shared_03/fsq/LongVideoBench',
                      help='Directory to save downloaded files')
    args = parser.parse_args()
    
    download_all_parts(args.download_dir)