import cv2
import numpy as np
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import timedelta



class FrameSelector:
    def __init__(self, conf):
        # llm
        self.llm = ChatOpenAI(
            api_key = conf.openai.GPT_API_KEY,
            model = conf.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = conf.openai.PROXY
        )

        self.visible_frames = None

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
    
    def inference(self, input):
        # input: 可见信息
        # output: 是否增加可见帧
        pass


if __name__ == "__main__":
    pass


    