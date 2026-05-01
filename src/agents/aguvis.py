from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.agents.parsers import parse_pyautogui_action, parse_aguvis_mobile_action
from src.utils import get_torch_dtype

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
        self.config.setdefault("user_message_template", USER_MSG_TEMPLATE)
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
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=self.config['device_map'])
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
    
    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        texts, all_images = [], []
        for screenshot, task in inputs:
            messages = self._get_chat_messages(screenshot, task)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
            text += "<|im_start|>assistant<|recipient|>os\n"
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            all_images.extend(image_inputs)

        batch_inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt")
        batch_inputs = batch_inputs.to(self.model.device)

        generated_ids = self.model.generate(**batch_inputs, temperature=self.config['temperature'], max_new_tokens=self.config['max_new_tokens'])
        trimmed = [out[len(inp):] for inp, out in zip(batch_inputs.input_ids, generated_ids)]
        output_texts = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

        return [self.postprocess(t.strip()) for t in output_texts]

    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        # prepare model inputs
        inputs = self._get_model_inputs(screenshot, task)
        inputs = inputs.to(self.model.device)
        # generate ids
        generated_ids = self.model.generate(**inputs, temperature=self.config['temperature'], max_new_tokens=self.config['max_new_tokens'])
        generated_ids_trimmed = generated_ids.tolist()[0][len(inputs.input_ids[0]) :]
        # decode ids
        output_text = self.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
        
        return self.postprocess(output_text)
    
    def postprocess(self, raw_output: str) -> AgentOutput:
        result = parse_pyautogui_action(raw_output)
        if result is not None:
            return result
        
        result = parse_aguvis_mobile_action(raw_output)
        if result is not None:
            return result
        
        logger.error(f"AGUVIS: unhandled action: {raw_output!r}")
        return AgentOutput(raw={"content": raw_output})
    
    def _get_model_inputs(self, screenshot: Image, task: str, history: str = "None"):
        messages = self._get_chat_messages(screenshot, task, history)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template)
        text += "<|im_start|>assistant<|recipient|>os\n"
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        return inputs
        
    def _get_chat_messages(self, screenshot: Image, task: str, history: str = "None"):
        system_message = { "role": "system", "content": self.config['grounding_system_message'] }
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
