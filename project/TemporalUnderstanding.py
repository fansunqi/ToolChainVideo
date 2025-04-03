import pdb
import os
import cv2
from PIL import Image
import math
import numpy as np
import datetime
import torch
import sqlite3
from torch.cuda.amp import autocast as autocast
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain.llms.openai import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
# from langchain import OpenAI, SQLDatabase
from langchain_community.utilities.sql_database import SQLDatabase


def format_seconds_to_time(seconds):
    # Convert seconds to a timedelta object
    time_delta = datetime.timedelta(seconds=seconds)

    # Extract the hours, minutes, and seconds components from the timedelta
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds // 60) % 60
    seconds = time_delta.seconds % 60

    # Use string formatting to create the formatted time string
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return formatted_time


class TemporalBase(object):
    def __init__(
        self,
        device="cpu",
        config = None,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        # ####caption initial
        self.config = config
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=self.torch_dtype
        ).to(self.device)

        self.llm = ChatOpenAI(
            api_key = self.config.openai.GPT_API_KEY,
            model = self.config.openai.GPT_MODEL_NAME,
            temperature = 0,
            base_url = self.config.openai.PROXY
        )

        ####other args
        self.frames = None
        self.fps = None
        self.video_len = None
        self.sql_path = None
        self.step = None
        self.sub_frames = None

    def inital_video(self, video_path, step=30, start_time=0, end_time=None):
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.step = step

        count = 0
        self.frames = []
        if end_time is None:
            end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                count += 1
                if count > (start_time * self.fps) and count <= (end_time * self.fps):
                    self.frames.append(frame)

                if count == (end_time * self.fps):
                    break
            else:
                break

        self.video_len = len(self.frames)
        self.sub_frames = self.frames[:: self.step]

    def reset_video(self):
        self.frames = None
        self.fps = None
        self.video_len = None
        self.sql_path = None
        self.step = None
        self.sub_frames = None

    def build_database(self, video_path):

        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        cursor.execute(
            "CREATE TABLE IF NOT EXISTS temporaldb (frame_id INTEGER PRIMARY KEY, frame_time TEXT)"
        )
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD visual_content TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD subtitles TEXT")
        conn.commit()

        cursor.execute("ALTER TABLE temporaldb ADD audio_content TEXT")
        conn.commit()

        cursor.execute("PRAGMA table_info(temporaldb)")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.commit()
        
        # pdb.set_trace()

        for id_i in range(0, len(self.sub_frames)):
            frame_id_i = id_i * self.step
            id_time_second = format_seconds_to_time(math.floor(frame_id_i / (self.fps)))
            cursor.execute(
                "INSERT INTO temporaldb (frame_id, frame_time) VALUES (?, ?)",
                (frame_id_i, id_time_second),
            )
            conn.commit()

        conn.close()

    def run_on_video(self, video_path, step=30, db_version=0):
        
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        self.sql_path = os.path.join(video_dir, video_name + "_" + str(db_version) + ".db")
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='temporaldb';"
        )
        rows = cursor.fetchall()
        
        # temporaldb 表为空才建表
        if len(rows) == 0:
            
            print("\nTemporaldb is empty, begin to build...")
            
            self.inital_video(video_path, step)
            
            self.build_database(video_path)
            self.run_VideoCaption()
            print("\nTemporaldb is built.")
        else:
            print("\nTemporaldb already exists.")

        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM temporaldb")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.close()

        return rows[0]

    # 下面这个函数并没有实际用处
    def run_on_question(self, question):
        ####visual
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM temporaldb")
        rows = cursor.fetchall()
        print("### Table temporaldb now is", rows)
        conn.close()

        db = SQLDatabase.from_uri("sqlite:///" + self.sql_path)
        db_chain = SQLDatabaseChain(llm=self.llm, database=db, top_k=100, verbose=True)
        result = db_chain.run(question)

        return result

    def run_VideoCaption(self):
        conn = sqlite3.connect(self.sql_path)
        cursor = conn.cursor()

        for id_i in range(0, len(self.sub_frames)):
            frame_id_i = id_i * self.step
            caption_text = self.caption_by_img(self.sub_frames[id_i])
            cursor.execute(
                "UPDATE temporaldb SET visual_content = ? WHERE frame_id = ?;",
                (caption_text, frame_id_i),
            )
            conn.commit()

        conn.close()

    def caption_by_img(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(rgb_image)
        inputs = self.processor(raw_image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer
