import re
import sys
sys.path.append("projects/Grounded-Video-LLM")
import pdb
import numpy as np
import torch
from omegaconf import OmegaConf
import decord
from decord import VideoReader

from models.llava_next_video import LLAVA_NEXT_VIDEO
from inference import parse_args
from mm_utils.video_utils import get_frame_indices
from mm_utils.utils import *
from datasets.chat.base_template import LLaMA3_Template, Vicuna_Template, Phi_3_5_Template, DEFAULT_IMAGE_TOKEN, GROUNDING_TOKEN

args = parse_args()


def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def parse_time_interval(text, duration, num_temporal_tokens=300, llm='phi3.5'):
    pattern = r"<(\d+)>"
    replaced_xs = []

    def replace_func(match):
        x = int(match.group(1))
        replaced_xs.append(x)
        m = duration * x / num_temporal_tokens
        if llm == 'phi3.5':
            return f" {m:.2f} seconds"
        elif llm == 'llama3':
            return f"{m:.2f} seconds"
        else:
            return f"{m:.2f} sec"  # fallback

    new_text = re.sub(pattern, replace_func, text)
    return new_text, replaced_xs



class TemporalGrounding:
    def __init__(
        self,
        conf = None, 
    ):
        
        self.visible_frames = None
        self.video_path = None
        
        self.llm_type = conf.tool.temporal_grounding.llm
        
        weight_path = conf.tool.temporal_grounding.weight_path
        config_path = f"{weight_path}/Phi-3.5-vision-instruct"
        tokenizer_path = f"{weight_path}/Phi-3.5-mini-instruct"
        pretrained_video_path = f"{weight_path}/internvideo/vision-encoder-InternVideo2-stage2_1b-224p-f4.pt"
        pretrained_vision_proj_llm_path = f"{weight_path}/Phi-3.5-vision-instruct-seperated"
        ckpt_path = f"{weight_path}/ckpt/sft_llava_next_video_phi3.5_mix_sft_multi_modal_projector_video_projecter_language_model.pth"
        
        self.device = conf.tool.temporal_grounding.device
        
        print("Loading Temporal-Grounding-Tool model...\n")
        self.model = LLAVA_NEXT_VIDEO(
            dtype=args.dtype, 
            stage=args.stage, 
            max_txt_len=args.max_txt_len, 
            num_frames=args.num_frames,  # NOTE
            num_segs=args.num_segs,      # NOTE
            num_temporal_tokens=args.num_temporal_tokens,
            lora=args.lora,
            llm=self.llm_type,
            attn_implementation=args.attn_implementation,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            pretrained_video_path=pretrained_video_path,
            pretrained_vision_proj_llm_path=pretrained_vision_proj_llm_path, 
        )
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        if 'multi_modal_projector' in ckpt.keys():
            self.model.multi_modal_projector.load_state_dict(ckpt['multi_modal_projector'])
        if 'video_projecter' in ckpt.keys():
            self.model.video_projecter.load_state_dict(ckpt['video_projecter'])
        if 'language_model' in ckpt.keys():
            self.model.language_model.load_state_dict(ckpt['language_model'])  
        self.model.eval()
        self.model.to(self.device)
        print("Finish loading Temporal-Grounding-Tool model.\n")
    
    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames  
    
    def set_video_path(self, video_path):
        self.video_path = video_path  
     
    def read_frames_decord(self, video_path):
        video_reader = VideoReader(video_path, num_threads=1)
        
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)
        
        frame_indices = get_frame_indices(
            num_frames = 96, 
            vlen = vlen, 
            sample = "middle", 
            fix_start = None,
            input_fps = fps,
            max_num_frames = -1
        )
        
        try:
            frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        except decord.DECORDError as e:
            print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
            print(f'decord.DECORDError报错: {e}')
        except Exception as e:
            print(f'解码错误: {video_path}, {vlen}, {fps}, {duration}')
            print(f'Exception报错: {e}')

        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

        return frames, frame_indices, float(fps), vlen, duration

    def create_inputs(self, video_path, prompt_grounding):
        video_processor = frame_transform(image_size=224, mean=INTERNVIDEO_MEAN, std=INTERNVIDEO_STD)
        image_processor = frame_transform(image_size=336, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        
        pixel_values, frame_indices, fps, total_frame_num, duration = self.read_frames_decord(video_path)
        
        temporal_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            temporal_pixel_values.append(video_processor(pixel_values[i]))
        temporal_pixel_values = torch.tensor(np.array(temporal_pixel_values)) # [num_frames, 3, 224, 224]
        temporal_pixel_values = temporal_pixel_values.unsqueeze(0)

        num_frames_per_seg = int(args.num_frames // args.num_segs)
        indices_spatial = [(i*num_frames_per_seg) + int(num_frames_per_seg/2) for i in range(args.num_segs)]
        spatial_pixel_values = []
        for i_spatial in indices_spatial:
            spatial_pixel_values.append(image_processor(pixel_values[i_spatial]))
        spatial_pixel_values = torch.tensor(np.array(spatial_pixel_values)) # [num_segs, 3, 336, 336]
        spatial_pixel_values = spatial_pixel_values.unsqueeze(0)
        
        chat_template = {'phi3.5': Phi_3_5_Template(), 'llama3': LLaMA3_Template(), 'vicuna': Vicuna_Template()}[self.llm_type]
        
        # TODO
        conv = [
            {"from": "human", "value": DEFAULT_IMAGE_TOKEN + ' ' + GROUNDING_TOKEN + '\n' + prompt_grounding},
            {"from": "gpt", "value": ''}                
        ]
        sep, eos = chat_template.separator.apply()
        prompt = chat_template.encode(conv).replace(eos, '')

        samples = {
            "video_ids": [video_path],
            "question_ids": [video_path],
            "prompts": [prompt],
            "temporal_pixel_values": temporal_pixel_values.to(self.device),
            "spatial_pixel_values": spatial_pixel_values.to(self.device),
        }
    
        return samples, duration
    
    @prompts(
        name = "temporal-grounding-tool",
        description = "Useful when you want to know which time segment of the video a certain event or content appears in"
        "The input to this tool must be a question without options, such as 'How many children are in the video?', instead of 'How many children are in the video? A. 1 B. 2 C. 3 D. 4'."
    )
    def inference(self, input):

        # TODO 对 input 进行清除选项

        prompt_grounding = f"Give you a textual query: '{input}'. When does the described content occur in the video? Please return the start and end timestamps."
        
        samples_grounding, duration_grounding = self.create_inputs(self.video_path, prompt_grounding)
        
        generate_kwargs = {
            "do_sample": args.do_sample,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "temperature":args.temperature,
            "top_p":args.top_p,
        }
        
        with torch.cuda.amp.autocast(enabled=True, dtype=self.model.dtype):
            with torch.inference_mode():
                pred_texts_grounding = self.model.generate(samples_grounding, **generate_kwargs)[0]
        
        # print('\n******grounding example******')
        # print(samples_grounding['prompts'][0])
        result, time_token_list = parse_time_interval(pred_texts_grounding, duration_grounding, args.num_temporal_tokens, self.llm_type)
        
        # 这里直接对 self.visible_frames 进行操作，添加帧
        assert len(time_token_list) == 2
        start_token_idx = time_token_list[0]
        end_token_idx = time_token_list[1]

        total_frames_num = self.visible_frames.video_info["total_frames"]
        start_frame_idx = int(start_token_idx / args.num_temporal_tokens * total_frames_num)
        if start_frame_idx < 0:
            start_frame_idx = 0
        end_frame_idx = int(end_token_idx / args.num_temporal_tokens * total_frames_num) + 1
        if end_frame_idx >= total_frames_num:
            end_frame_idx = total_frames_num - 1

        # 最小间隔为 1s 抽一帧
        minimal_interval = int(1 * self.visible_frames.video_info["fps"])
        frame_indices = range(start_frame_idx, end_frame_idx + 1, minimal_interval)

        self.visible_frames.add_frames(frame_indices=frame_indices)

        return result
        
        
        

if __name__ == "__main__":
    conf = OmegaConf.load("/home/fsq/video_agent/ToolChainVideo/config/nextqa_new_tool.yaml")
    temporal_grounding = TemporalGrounding(conf)
    
    # e.g.1
    video_path = "/home/fsq/video_agent/ToolChainVideo/projects/Grounded-Video-LLM/experiments/_3klvlS4W7A.mp4"
    input_question = "The female host wearing purple clothes is reporting news in the studio"
    
    temporal_grounding.set_video_path(video_path)
    result = temporal_grounding.inference(input=input_question)
    print(f"Result: {result}")
    
    print("main done") 