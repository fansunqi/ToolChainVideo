from datasets import load_dataset
from huggingface_hub import login
login(token="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")
# ds = load_dataset("longvideobench/LongVideoBench")

from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="longvideobench/LongVideoBench",
    repo_type="dataset",
    filename="videos.tar.part.aa"
)

print(f"Downloaded to: {file_path}")
