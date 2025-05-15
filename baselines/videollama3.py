import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor



device = "cuda:0"
model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


# start_time = time.time()  # 开始计时

# video_path = "/mnt/Shared_03/fsq/VideoMME/unzipped/data/_8lBR0E_Tx8.mp4"
# question = "What is this video about?"

def video_qa(video_path, question, frames_num):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": frames_num}},
                {"type": "text", "text": question},
            ]
        },
    ]

    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(response)
    
    return response


    # end_time = time.time()  # 结束计时
    # elapsed_time = end_time - start_time
    # print(f"Processing time: {elapsed_time:.2f} seconds")