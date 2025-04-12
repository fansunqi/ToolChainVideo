import cv2
from PIL import Image
from typing import List
import torch
import re
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



def clean_question(question):
    # 查找关键字并移除之后的内容
    keyword = "Choose your answer from below options:"
    if keyword in question:
        question = question.split(keyword)[0].strip()
    return question


class ImageQA:
    def __init__(
        self,
        conf = None, 
        device = "cuda:0"
    ):
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

        self.visible_frames = None

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames

    def image_qa(
        self,
        image,
        question
    ):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer

    @prompts(
        name = "image-question-answering-tool",
        description = "Useful when you need to ask something about the frames in the video."
        "The input to this tool must be a question without options, such as 'How many children are in the video?', instead of 'How many children are in the video? A. 1 B. 2 C. 3 D. 4'."
    )
    def inference(self, input):
        
        # 如果输入问题中包含选项，则去掉选项
        input = clean_question(input)
        
        result = "Here are the question-answering result of sampled frames:\n"
        for frame in self.visible_frames.frames:
            
            answer = self.image_qa(frame.image, input)
            add_info = f"Question: {input}\tAnswer: {answer}"
            
            if add_info not in frame.description:
                frame.description += (add_info + "\n")
                
            print(f"QA... Frame {frame.index}: {add_info}\n")
            result += f"\nFrame {frame.index}: {add_info}"

        return result       




if __name__ == "__main__":
    # 加载预训练的处理器和模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载图像
    image_path = "/home/fsq/video_agent/ToolChainVideo/misc/car.jpg"
    image = Image.open(image_path).convert("RGB")

    # 定义问题
    question = "What is the color of the car?"
    
    
    # 处理输入
    inputs = processor(image, question, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 推理
    output = model.generate(**inputs)

    # 解码答案
    answer = processor.decode(output[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
