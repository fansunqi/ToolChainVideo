import cv2
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
import os
import base64
import openai



def extract_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in tqdm(frame_indices, desc="抽取帧"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        frames.append(img)
    
    cap.release()
    return frames

def pil_image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def video_qa(client, frames, question):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": question}] + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_image_to_base64(img)}"
                    }
                } for img in frames
            ]
        }
    ]

    response = client.chat.completions.create(
        # model="gpt-4o",
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        # max_tokens=500,
        # temperature=0.2,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # === 输入部分 ===
    video_path = "/mnt/Shared_03/fsq/VideoMME/unzipped/data/_8lBR0E_Tx8.mp4"
    question = "视频里大概发生了什么？"
    N = 300  # 抽5帧

    # === 抽帧 ===
    frames = extract_frames(video_path, N)

    # === 问答 ===
    answer = video_qa(frames, question)

    print("\n=== GPT-4o 的回答 ===")
    print(answer)
