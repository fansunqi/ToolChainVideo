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

5. Download the checkpoints  and bulid related project🧩:

   - **download LLaVA for Image QA**
     
     git clone this [repo](https://github.com/haotian-liu/LLaVA), modify ```LLaVA/llava/eval/run_llava.py``` and install following instrutions.


## Tools

Thanks to the authors of these open-source projects for providing excellent projects.

- Object Tracker: 
    + YOLO by ultralytics: https://github.com/ultralytics/ultralytics
- Image Captioner: 
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
- Image QA:
    + BLIP: https://huggingface.co/docs/transformers/model_doc/blip
    + LLaVA: https://github.com/haotian-liu/LLaVA
- Frame Selector

## NExT-QA 试验

下载 NeXT-QA 数据：
```
git clone git@github.com:doc-doc/NExT-QA.git
```
specify your data path in ```config/nextqa.yaml```

运行指令：

```
python scripts/main.py
```

默认使用的 config 是 `config/nextqa.yaml`

评测指令：

```
python eval/eval_nextqa.py
```


(04.11 更新) 最新的运行指令：

```
python main_new_tools.py
```

默认使用的 config 是 `config/nextqa_new_tool.yaml`

最新的评测指令：

```
python eval.py
```