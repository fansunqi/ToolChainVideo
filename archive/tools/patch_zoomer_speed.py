import os
import cv2
import pdb
import time 
import base64
from openai import OpenAI
from pydantic import BaseModel
from omegaconf import OmegaConf

from prompts import PATCH_ZOOMER_PROMPT
from engine.openai import ChatOpenAI
from tools.common import image_resize_for_vlm


def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]


class PatchZoomer:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf
        self.perf_stats = {
            'llm_calls': [],
            'image_resize': [],
            'patch_extraction': [],
            'total_frame': []
        }

        model_string = self.conf.tool.patch_zoomer.vlm_gpt_model_name
        print(f"\nInitializing Patch Zoomer Tool with model: {model_string}")
        self.llm_engine = ChatOpenAI(
            model_string=model_string, 
            is_multimodal=True,
            enable_cache=self.conf.tool.patch_zoomer.use_cache
        )

        self.matching_dict = {
            "A": "top-left",
            "B": "top-right",
            "C": "bottom-left",
            "D": "bottom-right",
            "E": "center"
        }


    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
        
    def set_video_path(self, video_path):
        self.video_path = video_path

    def _get_patch(self, image, image_path, patch, save_path, zoom_factor=2):
        """Extract and save a specific patch from the image with 10% margins."""
        start_time = time.time()
        
        if image_path:
            img = cv2.imread(image_path)
        else:
            img = image
            
        height, width = img.shape[:2]
        
        quarter_h = height // 2
        quarter_w = width // 2
        
        margin_h = int(quarter_h * 0.1)
        margin_w = int(quarter_w * 0.1)
        
        patch_coords = {
            'A': ((max(0, 0 - margin_w), max(0, 0 - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, quarter_h + margin_h))),
            'B': ((max(0, quarter_w - margin_w), max(0, 0 - margin_h)),
                  (min(width, width + margin_w), min(height, quarter_h + margin_h))),
            'C': ((max(0, 0 - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, quarter_w + margin_w), min(height, height + margin_h))),
            'D': ((max(0, quarter_w - margin_w), max(0, quarter_h - margin_h)),
                  (min(width, width + margin_w), min(height, height + margin_h))),
            'E': ((max(0, quarter_w//2 - margin_w), max(0, quarter_h//2 - margin_h)),
                  (min(width, quarter_w//2 + quarter_w + margin_w), 
                   min(height, quarter_h//2 + quarter_h + margin_h)))
        }
        
        (x1, y1), (x2, y2) = patch_coords[patch]
        patch_img = img[y1:y2, x1:x2]
        
        zoomed_patch = cv2.resize(patch_img, 
                                (patch_img.shape[1] * zoom_factor, 
                                 patch_img.shape[0] * zoom_factor), 
                                interpolation=cv2.INTER_LINEAR)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, zoomed_patch)
        
        elapsed_time = time.time() - start_time
        self.perf_stats['patch_extraction'].append(elapsed_time)
        
        return zoomed_patch

    def patch_zoom_qa(self, question, image=None, image_path=None, save_path=None, zoom_factor=2):
        start_time = time.time()
        
        # 测量图像处理时间
        resize_start = time.time()
        prompt = PATCH_ZOOMER_PROMPT.format(question=question)
        if image_path:
            assert image is None, "both image and image path are not None"
            with open(image_path, 'rb') as file:
                image_bytes = file.read()
            input_data = [prompt, image_bytes]
        else:
            assert image is not None, "both image and image_path are None"
            image = image_resize_for_vlm(image)
            _, buffer = cv2.imencode(".jpg", image)
            input_data = [prompt, buffer]
        resize_time = time.time() - resize_start
        self.perf_stats['image_resize'].append(resize_time)
        
        # 测量LLM调用时间
        llm_start = time.time()
        response = self.llm_engine(input_data, response_format=PatchZoomerResponse)
        llm_time = time.time() - llm_start
        self.perf_stats['llm_calls'].append(llm_time)
        
        if image_path:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            image_name = "image"
        
        patch_info = []
        # for patch in response.patch:
        for patch in response["patch"]:
            patch_name = self.matching_dict[patch]

            if save_path:
                save_patch_path = os.path.join(save_path, f"{image_name}_{patch_name}_zoomed_{zoom_factor}x.png")
            else:
                save_patch_path = None
            
            zoomed_patch = self._get_patch(image, image_path, patch, save_patch_path, zoom_factor)
            
            patch_info.append({
                "path": save_patch_path,
                "description": f"The {self.matching_dict[patch]} region of the image: {image_path}.",
                "zoomed_patch": zoomed_patch
            })
        
        total_time = time.time() - start_time
        self.perf_stats['total_frame'].append(total_time)
        
        return {
            # "analysis": response.analysis,
            "analysis": response["analysis"],
            "patches": patch_info
        }


    @prompts(
        name = "relevant-patch-zooming-tool",
        description = "placeholder"
    )
    def inference(self, input):

        from concurrent.futures import ThreadPoolExecutor
        import concurrent.futures

        def process_frame(args):
            frame_idx, visible_frame = args
            result_dict = self.patch_zoom_qa(
                question=input,
                image=visible_frame.image,
                # save_path=...
            )
            patches = result_dict["patches"]
            if len(patches) == 1:
                return frame_idx, patches[0]["zoomed_patch"]
            return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            frame_args = [(idx, frame) for idx, frame in enumerate(self.visible_frames.frames)]
            future_to_frame = {executor.submit(process_frame, args): args for args in frame_args}
            
            for future in concurrent.futures.as_completed(future_to_frame):
                result = future.result()
                if result is not None:
                    frame_idx, zoomed_patch = result
                    self.visible_frames.frames[frame_idx].image = zoomed_patch

        # 打印性能统计
        print("\n=== Performance Analysis ===")
        print(f"Processed {len(self.visible_frames.frames)} frames")
        print("\nAverage times per frame:")
        print(f"- Image resize: {sum(self.perf_stats['image_resize'])/len(self.perf_stats['image_resize']):.3f}s")
        print(f"- LLM inference: {sum(self.perf_stats['llm_calls'])/len(self.perf_stats['llm_calls']):.3f}s")
        print(f"- Patch extraction: {sum(self.perf_stats['patch_extraction'])/len(self.perf_stats['patch_extraction']):.3f}s")
        print(f"- Total frame processing: {sum(self.perf_stats['total_frame'])/len(self.perf_stats['total_frame']):.3f}s")
        print("\nMax times:")
        print(f"- Image resize: {max(self.perf_stats['image_resize']):.3f}s")
        print(f"- LLM inference: {max(self.perf_stats['llm_calls']):.3f}s")
        print(f"- Patch extraction: {max(self.perf_stats['patch_extraction']):.3f}s")
        print(f"- Total frame processing: {max(self.perf_stats['total_frame']):.3f}s")
        print("========================")

        return "Performance analysis completed"

        



if __name__ == "__main__":

    conf = OmegaConf.load("/home/fsq/video_agent/ToolChainVideo/config/nextqa_st.yaml")
    patch_zoomer = PatchZoomer(conf)

    image_path = "/home/fsq/video_agent/Octotools-Video/src/tools/relevant_patch_zoomer/examples/car.png"
    question = "What is the color of the car?"
    save_path = "/home/fsq/video_agent/ToolChainVideo/misc/patch_zoomer_results"

    result = patch_zoomer.patch_zoom_qa(image_path=image_path, question=question, save_path=save_path)

# python -m tools.patch_zoomer