import cv2
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# 模型路径
model_path = "liuhaotian/llava-v1.5-7b"

# 加载预训练模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# 提示语和图像路径
prompt1 = "What are the things I should be cautious about when I visit here?"
prompt2 = "What are the things I should be cautious about when I visit here?"

image_file = "/home/fsq/video_agent/ToolChainVideo/misc/view.jpg"
image_file2 = "/home/fsq/video_agent/ToolChainVideo/misc/view_cabin.jpg"

# 使用 OpenCV 读取图像
image = cv2.imread(image_file)
image2 = cv2.imread(image_file2)

# 将 BGR 图像转换为 RGB（LLaVA 模型通常需要 RGB 格式）
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 将 OpenCV 图像转换为 PIL.Image 格式
pil_image = Image.fromarray(image)
pil_image2 = Image.fromarray(image2)

prompt_list = [prompt1, prompt2]
pil_image_list = [pil_image, pil_image2]

# 构造参数
args = type('Args', (), {
    "model_path": model_path, 
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "query": prompt_list,
    "conv_mode": None,
    "input_pil_image": pil_image_list,
    "image_file": None,  # 传递处理后的图像
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

# 调用 eval_model 进行推理
outputs = eval_model(args)
print(outputs)