import cv2
import os
import numpy as np
from openai import OpenAI
import time
import pdb
from omegaconf import OmegaConf
import time
from tools.common import openai_multimodal_qa


def prompts(name, description):
    
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

   
def sample_evenly(lst, num_samples):
    if num_samples <= 0:
        return []
    if num_samples >= len(lst):
        return lst[:]
    step = (len(lst) - 1) / (num_samples - 1)
    return [lst[int(round(i * step))] for i in range(num_samples)]

# 留白模式
def sample_adaptively_whitespace(lst):

    if len(lst) <= 2**2:
        grid_size = 2
    else:
        grid_size = 3

    num_samples = grid_size**2
    
    if num_samples >= len(lst):
        samples = lst[:]
    else:
        step = (len(lst) - 1) / (num_samples - 1)
        samples = [lst[int(round(i * step))] for i in range(num_samples)]

    return samples, grid_size

# 舍弃模式
def sample_adaptively_drop(lst):

    if len(lst) >= 3**2:
        grid_size = 3
    elif len(lst) >= 2**2:
        grid_size = 2
    else:
        grid_size = 1
    
    if grid_size > 1:
        num_samples = grid_size**2
        step = (len(lst) - 1) / (num_samples - 1)
        samples = [lst[int(round(i * step))] for i in range(num_samples)]
    else:
        samples = lst

    return samples, grid_size


def draw_grid_img(frames, grid_size, spacer=0, render_pos='topright'):

    frame_height, frame_width = frames[0].shape[:2]
    
    grid_height = grid_size * frame_height + (grid_size - 1) * spacer
    grid_width = grid_size * frame_width + (grid_size - 1) * spacer

    grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j
            if index < len(frames):
                frame = frames[index]
                cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
                max_dim = int(min(frame.shape[:2]) * 0.5)
                overlay = frame.copy()
                if render_pos == 'center':
                    circle_center = (cX, cY)
                else:
                    circle_center = (frame.shape[1] - max_dim // 2, max_dim // 2)
                cv2.circle(overlay, circle_center,
                        max_dim // 2, (255, 255, 255), -1)
                alpha = 0.3
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                cv2.circle(frame, circle_center, max_dim // 2, (255, 255, 255), 2)
                font_scale = max_dim / 50
                text_size = cv2.getTextSize(
                    str(index + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                if render_pos == 'center':
                    text_x = cX - text_size[0] // 2
                    text_y = cY + text_size[1] // 2
                else:
                    text_x = frame.shape[1] - text_size[0] // 2 - max_dim // 2
                    text_y = text_size[1] // 2 + max_dim // 2
                cv2.putText(frame, str(index + 1), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                y1 = i * (frame_height + spacer)
                y2 = y1 + frame_height
                x1 = j * (frame_width + spacer)
                x2 = x1 + frame_width
                grid_img[y1:y2, x1:x2] = frame
    
    return grid_img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized



class ImageGridQA:
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

        self.grid_size = conf.tool.image_grid_qa.init_grid_size

        self.save_path = conf.tool.image_grid_qa.save_path

        self.visible_frames = None
        self.video_path = None

    def set_frames(self, visible_frames):
        self.visible_frames = visible_frames
        
    def set_video_path(self, video_path):
        self.video_path = video_path  

    def select_grid_frames(self):

        if self.mode == "by_visible_frames":
            # create frame by_visible_frames
            sampled_visible_frames, grid_size = sample_adaptively_whitespace(self.visible_frames.frames)
            # 重置 self.grid_size
            self.grid_size = grid_size
            frames = []
            actual_indices = []
            for sampled_frame in sampled_visible_frames:
                frame = image_resize(sampled_frame.image, width=200)
                frames.append(frame)
                actual_indices.append(sampled_frame.index)
        
        elif self.mode == "by_video_path":
            # create frame by_video_path, 重新读取视频文件
            video = cv2.VideoCapture(self.video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            center_frame = int(total_frames / 2)
            num_frames = self.grid_size**2
            half_num_frames = num_frames // 2
            interval_frames = int(total_frames / num_frames - 1)
            frame_indices = [max(0,
                                min(center_frame + i * interval_frames,
                                    total_frames - 1)) for i in range(-half_num_frames,
                                                                    half_num_frames + 1)]
            frames = []
            actual_indices = []
            for index in frame_indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = video.read()
                if success:
                    frame = image_resize(frame, width=200)
                    frames.append(frame)
                    actual_indices.append(index)
                else:
                    print(f"Warning: Frame {index} not found")
                    print(f"Total frames: {total_frames}")
                    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = video.read()
                    frame = image_resize(frame, width=200)
                    frame = frame * 0
                    frames.append(frame)
                    actual_indices.append(index)
            video.release()
        
        else:
            raise ValueError("self.mode in image_grid_qa error.")
        
        return frames, actual_indices


    @prompts(
        name = "image-grid-qa-tool",
        description = "Useful when you want to know the whole event or action in the video. This tool arranges multiple images into an image grid, allowing the MLLM to analyze the events or actions taking place in the video."
        "The input to this tool must be a question, such as 'How many children are in the video?' "
    )
    def inference(self, input):

        frames, actual_indices = self.select_grid_frames()

        grid_img = draw_grid_img(frames, self.grid_size)

        if self.save_path:
            timestamp = time.strftime("%Y%m%d%H%M%S")
            output_img_path = os.path.join(self.save_path, f"grid_image_sample_{timestamp}.png")
            cv2.imwrite(output_img_path, grid_img)
            print(f"Image grid saved to {output_img_path}.")

        grid_num = self.grid_size**2

        prompt_image_grid_qa = (
            f"I will show an image sequence of {grid_num} sampled frames from a video. "
            f"I have annotated the images with numbered circles. "
            f"Based on the video, try to answer this question: "
            f"{input}"
        )

        result = openai_multimodal_qa(
            model_name=self.conf.tool.image_grid_qa.vlm_gpt_model_name,
            client=self.client_gpt,
            prompt=prompt_image_grid_qa,
            image=grid_img
        )
        
        return result

        # TODO: 要不要把这个结果加到 visible_frame.description 中去


if __name__ == "__main__":

    conf = OmegaConf.load("/home/fsq/video_agent/ToolChainVideo/config/nextqa_st.yaml")
    conf.tool.image_grid_qa.mode = "by_video_path"
    image_grid_qa = ImageGridQA(conf)

    video_path = "/share_data/NExT-QA/NExTVideo/1106/4010069381.mp4"
    question_w_options = "How do the two man play the instrument? Choose your answer from below options: A.roll the handle, B.tap their feet, C.strum the string, D.hit with sticks, E.pat with hand."
    

    image_grid_qa.set_video_path(video_path)
    
    result = image_grid_qa.inference(input=question_w_options)
    print(f"Result: {result}")
    
    print("main done")

# python -m tools.image_grid_qa 