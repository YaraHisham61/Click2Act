from typing import Literal
from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.agents.parsers import parse_pyautogui_action, parse_aguvis_mobile_action
from src.utils import get_torch_dtype

import json
from pathlib import Path
from loguru import logger
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor



class AGUVISAgent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)
        
        self.config.setdefault("temperature", 0)
        self.config.setdefault("max_new_tokens", 1024)
        self.config.setdefault("model_path", str(MODELS_PATH / "aguvis-7B-720P"))
        self.config.setdefault("repo_id", "xlangai/Aguvis-7B-720P")
        self.config.setdefault("dtype", "float16")
        self.config.setdefault("device_map", "auto")
        self.config.setdefault("grounding_system_message", GROUNDING_SYS_MSG)
        self.config.setdefault("agent_system_message", AGENT_SYS_MSG)
        self.config.setdefault("user_message_template", USER_MSG_TEMPLATE)
        self.config.setdefault("mode", "grounding") # ["self-plan", "force-plan", "grounding"]
        self.config.setdefault("low_level_instruction", None) # if there's detailed low level instructions to enforce instead of thinking
        # set dtype
        self.dtype = get_torch_dtype(self.config['dtype'])
        logger.debug(f"initalize agent with config={config}")
        # download model if not exist
        self.model_path = Path(self.config["model_path"])
        if not self.model_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                local_dir=str(self.config["model_path"]),
                local_dir_use_symlinks=False,
            )
     
    def load(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=self.config['device_map'])
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        
    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        return self.predict_action_batch(inputs, grounding=True) # same as predict click but with grounding message
    
    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        return self.predict_action(screenshot, task, grounding=True) # same as predict click but with grounding message
    
    # ------------- action ----------------
    def predict_action(self, screenshot: Image, task: str, grounding=False) -> AgentOutput:
        return self.predict_action_batch([screenshot, task], grounding)[0]
    
    def predict_action_batch(self, inputs: list[tuple[Image, str]], grounding=False) -> list[AgentOutput]:
        system_message = self.config['grounding_system_message'] if grounding else self.config['agent_system_message'] 
        texts, all_images = [], []
        for screenshot, task in inputs:
            image_inputs, _, text = self._get_model_inputs(screenshot, task, system_message)
            all_images.extend(image_inputs)
            texts.append(text)

        batch_inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**batch_inputs, temperature=self.config['temperature'], max_new_tokens=self.config['max_new_tokens'])
        trimmed = [out[len(inp):] for inp, out in zip(batch_inputs.input_ids, generated_ids)]
        output_texts = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

        return [self.postprocess(t.strip()) for t in output_texts]

    
    def postprocess(self, raw_output: str) -> AgentOutput:
        result = parse_pyautogui_action(raw_output)
        if result is not None:
            return result
        
        result = parse_aguvis_mobile_action(raw_output)
        if result is not None:
            return result
        
        logger.error(f"AGUVIS: unhandled action: {raw_output!r}")
        return AgentOutput(raw={"content": raw_output})
    
    def _get_model_inputs(self, screenshot: Image, task: str, system_message: str, history: str = "None"):
        messages = self._get_chat_messages(screenshot, task, system_message, history)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
        # If low-level instruction is provided,  We enforce using "Action: {low_level_instruction} to guide generation"
        if self.config['low_level_instruction']:
            text += f"<|im_start|>assistant<|recipient|>all\nAction: {self.config['low_level_instruction']}\n"
        elif self.config['mode'] == "grounding":
            text += "<|im_start|>assistant<|recipient|>os\n"
        elif self.config['mode'] == "self-plan":
            text += "<|im_start|>assistant<|recipient|>"
        elif self.config['mode'] == "force-plan":
            text += "<|im_start|>assistant<|recipient|>all\nThought: "
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        return image_inputs, video_inputs, text
        
    def _get_chat_messages(self, screenshot: Image, task: str, system_message: str, history: str = "None"):
        system_message = { "role": "system", "content": system_message }
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot},
                {
                    "type": "text",
                    "text": self.config['user_message_template'].format( overall_goal=task, previous_actions=history),
                }
            ],
        }
        messages = [system_message, user_message]
        
        
        return  messages
        
# --- Prompts Ref: are taking from Aguvis code aguvis/src/constants.py
GROUNDING_SYS_MSG = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."

USER_MSG_TEMPLATE = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal}

Previous actions:
{previous_actions}
"""
# Chat Template
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

AGENT_SYS_MSG = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""
