from huggingface_hub import login
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    repo_type="model",
    filename="merges.txt",
)

print(f"Downloaded to: {file_path}")