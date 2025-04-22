import cv2
import numpy as np
from typing import List
from langchain_openai import ChatOpenAI
from prompts import SELECT_FRAMES_PROMPT
from pydantic import BaseModel, Field
from pprint import pprint
import pdb

def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


class Segment(BaseModel):
    segment_id: int = Field(description="The index of the video segment, start from 0")
    start_frame_idx: int = Field(description="The start frame index of the video segment")
    end_frame_idx: int = Field(description="The end frame index of the video segment")


class SegmentList(BaseModel):
    segments: List[Segment] = Field(description="A list of segments")


class FrameSelector:
    def __init__(self, conf):
        
        self.llm = ChatOpenAI(
            api_key = conf.openai.GPT_API_KEY,
            model = conf.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = conf.openai.PROXY
        )

        self.visible_frames = None
        self.video_path = None
    
    def set_video_path(self, video_path):
        self.video_path = video_path  
        
    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    

    @prompts(
        name = "frame-extraction-tool",
        description = "Useful when you find that the currently sampled frames do not provide enough information and more frames need to be extracted from the video to answer the question."
        "The input to this tool must be a question about the video that remains unresolved. For example, 'How many children are in the video? Choose your answer from below selections: A.one, B.three, C.seven, D.two, E.five.'",
    )
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
        segments_list = structured_llm.invoke(select_frames_prompt)

        print("segments_list: ", segments_list)

        # 扩展帧
        # 最小间隔单位是 fps
        fps = int(self.visible_frames.video_info["fps"])
        add_frames_indices_all = []
        invisible_segments_list = self.visible_frames.get_invisible_segments()
        for segment in segments_list.segments:
            invisible_segment = invisible_segments_list[segment.segment_id]
            if invisible_segment[0] == segment.start_frame_idx and invisible_segment[1] == segment.end_frame_idx:
                if segment.end_frame_idx - segment.start_frame_idx >= fps:
                    add_frames_indices = range(segment.start_frame_idx, segment.end_frame_idx, fps)
                    print("add_frames_indices: ", add_frames_indices)
                    add_frames_indices_all.extend(add_frames_indices)

        add_frames_indices_all = list(set(add_frames_indices_all))
        add_frames_indices_all.sort()
        print("add_frames_indices_all: ", add_frames_indices_all)

        self.visible_frames.add_frames(frame_indices=add_frames_indices_all)

        return_message = "A series of potentially relevant frames have been successfully extracted and added to the visible frame set. Please continue by using other tools to analyze these newly added frames."
        return return_message



        


        








    