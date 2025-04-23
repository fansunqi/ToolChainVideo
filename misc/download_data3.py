from huggingface_hub import login
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="lmms-lab/Video-MME",
    repo_type="dataset",
    filename="videos_chunked_03.zip"
)

print(f"Downloaded to: {file_path}")