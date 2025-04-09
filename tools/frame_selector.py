import cv2
import numpy as np
from typing import List
from langchain_openai import ChatOpenAI
from prompts import SELECT_FRAMES_PROMPT
from pydantic import BaseModel, Field
from pprint import pprint


class Segment(BaseModel):
    segment_id: int = Field(description="The index of the video segment, start from 0")
    start_frame_idx: int = Field(description="The start frame index of the video segment")
    end_frame_idx: int = Field(description="The end frame index of the video segment")

class SegmentList(BaseModel):
    segments: List[Segment] = Field(description="A list of segments")

class FrameSelector:
    def __init__(self, conf=None):
        # llm
        # self.llm = ChatOpenAI(
        #     api_key = conf.openai.GPT_API_KEY,
        #     model = conf.openai.GPT_MODEL_NAME,
        #     temperature = 0,
        #     base_url = conf.openai.PROXY
        # )

        self.llm = ChatOpenAI(
            api_key="sk-lAWdJVGgMJikTuhW2PBIgwecI6Gwg0gdM3xKVxwYDiOW98ra",
            model="gpt-4o",
            temperature = 0,
            base_url="https://api.juheai.top/v1",
        )

        self.visible_frames = None

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    
    def inference(self, input):
        visible_info = self.visible_frames.get_frame_descriptions()

        select_frames_prompt = SELECT_FRAMES_PROMPT.format(
            num_frames = self.visible_frames.video_info["total_frames"],
            fps = self.visible_frames.video_info["fps"],
            visible_frames_info = self.visible_frames.get_frame_descriptions(),
            question = input,
            candidate_segment = self.visible_frames.invisible_segments_to_description()
        )

        structured_llm = self.llm.with_structured_output(SegmentList)
        output = structured_llm.invoke(select_frames_prompt)

        pprint(output)

        








    