import cv2
from PIL import Image
from typing import List
import torch
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)



def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator



class ImageCaptioner:
    def __init__(
        self,
        device="cuda:0"
    ):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=self.torch_dtype
        ).to(self.device)

    def set_frames(self, frames: List[cv2.Mat]):
        self.frames = frames


    def caption_image(
        self,
        image
    ):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        inputs = self.processor(raw_image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer



    @prompts(
        name = "image-caption-tool",
        description = "Useful when you need to caption the frames in the video."
        "The input to this tool is a placeholder and does not affect the tool's output."
    )
    def inference(self, input):
        result = "Here are the captions of frames:"
        for frame_idx, frame in enumerate(self.frames):
            frame_caption = self.caption_image(frame)
            print(f"Caption: {frame_caption}")
            result += f"\nFrame {frame_idx}: {frame_caption}"

        return result

    

if __name__ == "__main__":
    # e.g.1
    video_path = "/share_data/NExT-QA/NExTVideo/1164/3238737531.mp4"
    question_w_options = "How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five."

    image_captioner = ImageCaptioner()
    
    video_stride = 30  # 设置视频 stride，跳过的帧数
    from frame_selector import *
    frames = select_frames(video_path=video_path, video_stride=video_stride)
    image_captioner.set_frames(frames=frames)

    result_message = image_captioner.inference(input=question_w_options)
    print(result_message)          

