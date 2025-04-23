# from datasets import load_dataset
from huggingface_hub import login, hf_hub_download
import string
import os

# Login to Hugging Face
login(token="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")

def download_all_parts():
    # Create list of all suffixes: ab, ac, ..., az, ba, bb, bc, bd, be
    suffixes = []
    # First add ab through az (skipping aa)
    suffixes.extend([f"a{c}" for c in string.ascii_lowercase[1:]])  # Start from 'b'
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
                filename=filename
            )
            print(f"Downloaded {filename} to: {file_path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_all_parts()