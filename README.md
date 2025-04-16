# ToolChainVideo


## Setup and Configuration 🛠️

1. Clone the repository 📦:
   ```python
   git clone git@github.com:fansunqi/ToolChainVideo.git
   cd ToolChainVideo
   ```
2. Create a virtual environment 🧹 and install the dependencies 🧑‍🍳:
   ```python
   conda create -n tcv python=3.9
   conda activate tcv
   pip install -r requirements.txt
   ```
3. Set up your API key 🗝️ in `config/*.yaml`:
     ```python
     openai:
       GPT_API_KEY: "put your openai api key here"
       PROXY: "put your openai base url here"
     ```

5. Bulid related projects 🧩:

   - **download LLaVA for Image QA**
     
     Clone this [repo](https://github.com/haotian-liu/LLaVA), modify ```LLaVA/llava/eval/run_llava.py``` and install as instrutions.

     If you encounter this error:
     `TypeError: forward() got an unexpected keyword argument 'cache_position'`, 
     fix by add `cache_position=None` to the `forward()` method in `Class LlavaLlamaForCausalLM` as mentioned [here](https://github.com/huggingface/transformers/issues/29426).


## Tools

Thanks to the authors of these open-source projects for providing excellent projects.

Temporal Tools:
- Frame Selector
    + select frames of interest based on current information, driven by LLM.
- Temporal Grounding
    + Grounded-Video-LLM: https://github.com/WHB139426/Grounded-Video-LLM
- Temporal Refering
    + Grounded-Video-LLM: https://github.com/WHB139426/Grounded-Video-LLM
- Temporal QA
    + Grounded-Video-LLM: https://github.com/WHB139426/Grounded-Video-LLM
- 时序动作定位(TODO)
    + MMaction

Spatial Tools:
- Object Tracking 
    + YOLO by ultralytics: https://github.com/ultralytics/ultralytics
- Image Captioning
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
- Image QA
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
    + LLaVA: https://github.com/haotian-liu/LLaVA
- Action Recognition
    + MMAction: https://mmaction2.readthedocs.io/


Generalist Solution:
- Video QA
    + Qwen-VL-2.5: https://github.com/QwenLM/Qwen2.5-VL


## Download Datasets
- NeXT-QA：
  ```
  git clone git@github.com:doc-doc/NExT-QA.git
  ```
  specify your data path in ```config/nextqa.yaml```

## Run and evaluate

Specify your config file and run:
```
python main_new_tools.py
```

Evaluate:

```
python eval.py
```