from omegaconf import OmegaConf
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# TODO 将 qwen 环境合并至 tcv2 环境中，主要是 transformers 需要使用旧的


class VideoQA:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
        
    def set_video_path(self, video_path):
        self.video_path = video_path
        
    
    def video_qa(self, prompt_videoqa):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": self.video_path,
                    },
                    {
                        "type": "text", 
                        "text": prompt_videoqa,
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output_text[0]
        return result

    def inference(self, input):
        result = self.video_qa(prompt_videoqa=input)
        return result
        







if __name__ == "__main__":
    conf = OmegaConf.load("/home/fsq/video_agent/ToolChainVideo/config/nextqa.yaml")
    video_qa = VideoQA(conf)
    
    video_path = "/home/fsq/video_agent/ToolChainVideo/projects/Grounded-Video-LLM/experiments/_3klvlS4W7A.mp4"
    prompt_videoqa = "Question: What does this TV news report about?\nOptions:\n(A) thievery\n(B) community violence incidents\n(C) fashion show\n(D) aging population"
    # prompt_videoqa = "What does this TV news report about?"
    video_qa.set_video_path(video_path)
    
    result = video_qa.inference(input=prompt_videoqa)
    print(f"Result: {result}")
    
    print("main done")