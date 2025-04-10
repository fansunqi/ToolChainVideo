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
        conf = None, 
        device = "cuda:0"
    ):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=self.torch_dtype
        ).to(self.device)

        self.visible_frames = None

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames

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
        for frame in self.visible_frames.frames:
            frame_caption = self.caption_image(frame.image)
            
            print(f"Cpationing... Frame {frame.index}: {frame_caption}")
            result += f"\nFrame {frame.index}: {frame_caption}"

            frame.description += f"BLIP Caption: {frame_caption}\n"

        return result       

