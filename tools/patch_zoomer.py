import os
from openai import OpenAI
from pydantic import BaseModel
from omegaconf import OmegaConf

from prompts import PATCH_ZOOMER_PROMPT
from engine.openai import ChatOpenAI


class PatchZoomerResponse(BaseModel):
    analysis: str
    patch: list[str]


class PatchZoomer:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf

        self.mode = conf.tool.image_grid_qa.mode

        model_string = self.conf.tool.patch_zoomer.vlm_gpt_model_name
        print(f"\nInitializing Patch Zoomer Tool with model: {model_string}")
        self.llm_engine = ChatOpenAI(model_string=model_string, is_multimodal=True) if model_string else None

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

    def patch_zoom_qa(self, image, question):

        # Read image and create input data
        with open(image, 'rb') as file:
            image_bytes = file.read()
        prompt = PATCH_ZOOMER_PROMPT.format(question=question)
        input_data = [prompt, image_bytes]
        
        # Get response from LLM
        response = self.llm_engine(input_data, response_format=PatchZoomerResponse)
        
        # Save patches
        # image_dir = os.path.dirname(image)
        # image_name = os.path.splitext(os.path.basename(image))[0]
        
        # Update the return structure
        patch_info = []
        for patch in response.patch:
            patch_name = self.matching_dict[patch]
            # save_path = os.path.join(self.output_dir, 
            #                         f"{image_name}_{patch_name}_zoomed_{zoom_factor}x.png")
            # saved_path = self._save_patch(image, patch, save_path, zoom_factor)
            # save_path = os.path.abspath(saved_path)
            patch_info.append({
                # "path": save_path,
                "description": f"The {self.matching_dict[patch]} region of the image: {image}."
            })
        
        print(response.analysis)
        print(patch_info)

        return {
            "analysis": response.analysis,
            "patches": patch_info
        }
        



if __name__ == "__main__":

    conf = OmegaConf.load("/home/fsq/video_agent/ToolChainVideo/config/nextqa_st.yaml")
    patch_zoomer = PatchZoomer(conf)

    image_path = "/home/fsq/video_agent/Octotools-Video/src/tools/relevant_patch_zoomer/examples/car.png"
    question = "What is the color of the car?"

    result = patch_zoomer.patch_zoom_qa(image=image_path, question=question)

