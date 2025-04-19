
from openai import OpenAI



class PatchZoomer:
    def __init__(
        self,
        conf = None, 
    ):
        self.conf = conf

        self.mode = conf.tool.image_grid_qa.mode

        self.client_gpt = OpenAI(
            api_key = conf.openai.GPT_API_KEY,
            base_url = conf.openai.PROXY
        )

        self.prompt = f"""
Analyze this image to identify the most relevant region(s) for answering the question:

Question: {question}

The image is divided into 5 regions:
- (A) Top-left quarter
- (B) Top-right quarter
- (C) Bottom-left quarter
- (D) Bottom-right quarter
- (E) Center region (1/4 size, overlapping middle section)

Instructions:
1. First describe what you see in each of the five regions.
2. Then select the most relevant region(s) to answer the question.
3. Choose only the minimum necessary regions - avoid selecting redundant areas that show the same content. For example, if one patch contains the entire object(s), do not select another patch that only shows a part of the same object(s).


Response format:
<analysis>: Describe the image and five patches first. Then analyze the question and select the most relevant patch or list of patches.
<patch>: List of letters (A-E)
"""


    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
        
    def set_video_path(self, video_path):
        self.video_path = video_path

    def patch_zoom_qa(self, image_path, question):

        with open(image_path, 'rb') as file:
            image_bytes = file.read()
        prompt = self.prompt.format(question=question)
        input_data = [prompt, image_bytes]




if __name__ == "__main__":

    image_path = "/home/fsq/video_agent/Octotools-Video/src/tools/relevant_patch_zoomer/examples/car.png"
