from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.utils import get_torch_dtype

import json
import torch
from pathlib import Path
from loguru import logger
from PIL.Image import Image, Resampling
from qwen_vl_utils import smart_resize
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.agents.holo2.parser import ClickCoordinates, NavigationStep, parse_holo2_action

class HOLO2Agent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)
        
        self.config.setdefault("max_new_tokens", 32)
        self.config.setdefault("model_path", str(MODELS_PATH / "holo2-4b"))
        self.config.setdefault("repo_id", "Hcompany/Holo2-4B")
        self.config.setdefault("dtype", "float16")
        self.config.setdefault("device_map", "auto")
        self.config.setdefault("mode", "grounding") # [grounding, navigation]
        # set dtype
        self.dtype = get_torch_dtype(self.config['dtype'])
        # download model if not exist
        self.model_path = Path(self.config["model_path"])
        if not self.model_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                local_dir=str(self.config["model_path"]),
                local_dir_use_symlinks=False,
            )
     
    def load(self):
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=self.config['device_map'])
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        # REFINED [old]: default right-padding → [new]: left-padding required for decoder-only batch generation
        self.processor.tokenizer.padding_side = 'left'
    
    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        return self.predict_action_batch(inputs, grounding=True) # same as predict click but with grounding message
    
    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        return self.predict_action(screenshot, task, grounding=True) # same as predict click but with grounding message
    
    # ------------- action ----------------
    def predict_action(self, screenshot: Image, task: str, grounding=False) -> AgentOutput:
        return self.predict_action_batch([screenshot, task], grounding)[0]
    
    def predict_action_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        system_message = GROUNDING_MSG if self.config['mode'] == "grounding" else NAVIGATION_SYSTEM_PROMPT
        texts, all_images = [], []
        for screenshot, task in inputs:
            screenshot_processed = self.preprocess(screenshot)
            messages = self._get_chat_messages(screenshot_processed, task, system_message)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False)
            texts.append(text)
            all_images.append(screenshot_processed)

        batch_inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt")
        batch_inputs = batch_inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**batch_inputs, max_new_tokens=self.config['max_new_tokens'])
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(batch_inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        return [self.postprocess_grounding(t.strip()) for t in output_texts]

    def preprocess(self, screenshot: Image) -> Image:
        image_processor_config = self.processor.image_processor
        resized_height, resized_width = smart_resize(
            screenshot.height,
            screenshot.width,
            factor=image_processor_config.patch_size * image_processor_config.merge_size,
            min_pixels=image_processor_config.size.get("shortest_edge", None),
            max_pixels=image_processor_config.size.get("longest_edge", None),
        )

        processed_image = screenshot.resize(size=(resized_width, resized_height), resample=Resampling.LANCZOS)
        return processed_image
    
    def postprocess_grounding(self, raw_output: str) -> AgentOutput:
        try:
            output = parse_holo2_action(raw_output)
            if not output:
                raise Exception("couldn't parse output to one of actions")
        except Exception as err:
            logger.error(f"HOLO2: This action not handled to be parsed yet: raw_output={raw_output}\n error {err}")
            return AgentOutput(raw = {"content": raw_output})
    
    def _get_chat_messages(self, screenshot: Image, task: str, system_message: str, history: str = "None") -> list[dict[str, any]]:
        """Create the prompt structure for navigation task"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": f"{system_message}\t{task}"},
                ],
            },
        ]
        
# --- Prompts Ref: are taking from Holo2
GROUNDING_MSG = (
        "Localize an element on the GUI image according to the provided target and output a click position.\n"
        f"* You must output a valid JSON following the format: {ClickCoordinates.model_json_schema()}\n"
        "Your target is:\n"
)

# --- Nagivation: Ref from Holo https://github.com/hcompai/hai-cookbook/blob/main/utils/navigation.py
NAVIGATION_SYSTEM_PROMPT: str = f"""Imagine you are a robot browsing the web, just like humans. Now you need to complete a task.
In each iteration, you will receive an Observation that includes the last  screenshots of a web browser and the current memory of the agent.
You have also information about the step that the agent is trying to achieve to solve the task.
Carefully analyze the visual information to identify what to do, then follow the guidelines to choose the following action.
You should detail your thought (i.e. reasoning steps) before taking the action.
Also detail in the notes field of the action the extracted information relevant to solve the task.
Once you have enough information in the notes to answer the task, return an answer action with the detailed answer in the notes field.
This will be evaluated by an evaluator and should match all the criteria or requirements of the task.

Guidelines:
- store in the notes all the relevant information to solve the task that fulfill the task criteria. Be precise
- Use both the task and the step information to decide what to do
- if you want to write in a text field and the text field already has text, designate the text field by the text it contains and its type
- If there is a cookies notice, always accept all the cookies first
- The observation is the screenshot of the current page and the memory of the agent.
- If you see relevant information on the screenshot to answer the task, add it to the notes field of the action.
- If there is no relevant information on the screenshot to answer the task, add an empty string to the notes field of the action.
- If you see buttons that allow to navigate directly to relevant information, like jump to ... or go to ... , use them to navigate faster.
- In the answer action, give as many details a possible relevant to answering the task.
- if you want to write, don't click before. Directly use the write action
- to write, identify the web element which is type and the text it already contains
- If you want to use a search bar, directly write text in the search bar
- Don't scroll too much. Don't scroll if the number of scrolls is greater than 3
- Don't scroll if you are at the end of the webpage
- Only refresh if you identify a rate limit problem
- If you are looking for a single flights, click on round-trip to select 'one way'
- Never try to login, enter email or password. If there is a need to login, then go back.
- If you are facing a captcha on a website, try to solve it.

- if you have enough information in the screenshot and in the notes to answer the task, return an answer action with the detailed answer in the notes field
- The current date is 2026-05-01 14:16:03.

# <output_json_format>
# ```json
# {NavigationStep.model_json_schema()}
# ```
# </output_json_format>

"""
