from src.constants import MODELS_PATH
from src.utils import get_torch_dtype
from src.agents.base import GUIAgent, AgentOutput
from src.agents.parsers import parse_pyautogui_action, parse_aguvis_mobile_action
from src.agents.ui_zoomer.utils import run_ui_zoomer_flow, UIZoomerInput, get_points_from

import json
from pathlib import Path
from loguru import logger
from PIL.Image import Image
from vllm import LLM, SamplingParams
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

class AGUVISZoomerAgent(GUIAgent):
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
        # -- vllm config ---
        self.config.setdefault("tensor_parallel_size", 2) # 2 for kaggle T4x2
        self.config.setdefault("max_model_len", 4089) 
        self.config.setdefault("gpu_memory_utilization", 0.85) # Leave 15% VRAM for PyTorch overhead
        # --- ui zoom config ---
        self.config.setdefault("allow_ui_zoom", True)
        self.config.setdefault("ui_zoom_num_candidates", 8)
        self.config.setdefault("ui_zoom_point_var_ratio", 0.02) # click has 20 pixel radius for 1000 side
        self.config.setdefault("ui_zoom_confidence_threshold", 0.65)
        self.config.setdefault("ui_zoom_min_crop", 256)
        self.config.setdefault("ui_zoom_scale", 2.5) 
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
        self.model =  LLM(
            model=self.config['model_path'],
            max_model_len=self.config['max_model_len'],
            tensor_parallel_size=self.config['tensor_parallel_size'],
            gpu_memory_utilization=self.config['gpu_memory_utilization'],
            dtype="half",                   # TODO: configure based on dtype
            enforce_eager=True,             # Bypasses CUDA graph compilation errors
            trust_remote_code=True,      
            disable_custom_all_reduce=True, # Prevents custom communication kernels that crash T4s
        )
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        
        self.ignored_tokens: set[int] = set(self.tokenizer.encode("pyautogui.click(x=, y=)") + self.tokenizer.encode("pyautogui.doubleClick(x=, y=)") + [151658])
        
    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        if not self.config['allow_ui_zoom']:
            return self.predict_action_batch(inputs, grounding=True) # same as predict click but with grounding message
        # if ui zoom enabled
        try:
            return self.predict_click_batch_ui_zoom(inputs)
        except Exception as err:
            logger.error(f"Aguvis Ui Zoom Batch: Error\n error {err}")
            return [AgentOutput(raw={"error": str(err)})] * len(inputs)
    
    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        if not self.config['allow_ui_zoom']:
            return self.predict_action(screenshot, task, grounding=True) # same as predict click but with grounding message
        # if ui zoom enabled
        try:
            return self.predict_click_ui_zoom(screenshot, task)
        except Exception as err:
            logger.error(f"Aguvis Ui Zoom: Error\n error {err}")
            return AgentOutput(raw = {"error": str(err)})

    # ------------- action ----------------
    def predict_action(self, screenshot: Image, task: str, grounding=False) -> AgentOutput:
        return self.predict_action_batch([screenshot, task], grounding)[0]
    
    def predict_action_batch(self, inputs: list[tuple[Image, str]], grounding=False, sampling_params=None) -> list[AgentOutput]:
        system_message = self.config['grounding_system_message'] if grounding else self.config['agent_system_message']
        nsamples = len(inputs)
        
        if not sampling_params:
            sampling_params = [SamplingParams( temperature=self.config['temperature'], max_tokens=self.config['max_new_tokens'])] * nsamples
        
        vllm_inputs = []
        for screenshot, task in inputs:
            _, _, text = self._get_model_inputs(screenshot, task, system_message)
            vllm_inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": screenshot},
            })

        outputs = self.model.generate(vllm_inputs, sampling_params)
        output_texts = [out.outputs[0].text.strip() for out in outputs]

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
    
    def predict_click_ui_zoom(self,  screenshot: Image, task: str):
        w, h = screenshot.size
        sampling_params = SamplingParams(temperature=0.9, logprobs=1, n=self.config['ui_zoom_num_candidates'], max_tokens=self.config['max_new_tokens'])
        vllm_inputs = [{
            "prompt": self._get_model_inputs(screenshot, task, self.config['grounding_system_message'])[2],
            "multi_modal_data": {"image": screenshot},
        }]
        
        outputs = self.model.generate(vllm_inputs, sampling_params)
        output_texts = [raw_out.text.strip() for raw_out in outputs[0].outputs]
        
        ui_zoomer_output = run_ui_zoomer_flow(UIZoomerInput(
            model_outputs=outputs,
            w=w, h=h,
            ignored_tokens=self.ignored_tokens,
            point_var_ration=self.config['ui_zoom_point_var_ratio'],
            confidence_threshold=self.config['ui_zoom_confidence_threshold'],
            min_side=self.config['ui_zoom_min_crop'],
            scale=self.config['ui_zoom_scale'],
        ))
        if ui_zoomer_output.use_voted_point:
            return AgentOutput(
                coordinate=(ui_zoomer_output.voted_point[0] / w, ui_zoomer_output.voted_point[1] / h), # voted point normalize by w,h
                action_type='click',
                raw={"original_model_outputs": output_texts, "ui_zoomer_output": ui_zoomer_output.model_dump()}
            )
        # rerun on cropped image if not use_voted_point
        # crop image
        cropped_image = screenshot.copy().crop(ui_zoomer_output.bbox)
        wc, hc = cropped_image.size
        # generate output
        sampling_params = SamplingParams(temperature=1, max_tokens=self.config['max_new_tokens'])
        vllm_inputs = [{
            "prompt": self._get_model_inputs(cropped_image, task, self.config['grounding_system_message'])[2],
            "multi_modal_data": {"image": cropped_image},
        }]
        cropped_outputs = self.model.generate(vllm_inputs, sampling_params)
        cropped_output_texts = [raw_out.text.strip() for raw_out in cropped_outputs[0].outputs]
        
        cropped_point = get_points_from(cropped_outputs, wc, hc)[0] # multiple by wc, hc
        x_pred = cropped_point[0] + ui_zoomer_output.bbox[0] # x*wc + x0
        y_pred = cropped_point[1] + ui_zoomer_output.bbox[1] # y*hc + y0

        return AgentOutput(
            coordinate=(x_pred / w, y_pred / h), # voted point normalize by w,h
            action_type='click',
            raw={"original_model_outputs": output_texts, "cropped_model_outputs": cropped_output_texts, "ui_zoomer_output": ui_zoomer_output.model_dump()}
        )

    def predict_click_batch_ui_zoom(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        N = len(inputs)
        screenshots = [s for s, _ in inputs]
        tasks = [t for _, t in inputs]
        results = [None] * N

        # === Phase 1 ===
        sampling_params = SamplingParams(temperature=0.9, logprobs=1, n=self.config['ui_zoom_num_candidates'], max_tokens=self.config['max_new_tokens'])
        vllm_inputs = [{
            "prompt": self._get_model_inputs(s, t, self.config['grounding_system_message'])[2],
            "multi_modal_data": {"image": s},
        } for s, t in inputs]

        outputs = self.model.generate(vllm_inputs, [sampling_params] * N)
        output_texts_list = [[raw_out.text.strip() for raw_out in outputs[i].outputs] for i in range(N)]

        needs_stage2 = []  # [(i, ui_zoomer_output, w, h), ...]

        for i in range(N):
            w, h = screenshots[i].size
            try:
                ui_zoomer_output = run_ui_zoomer_flow(UIZoomerInput(
                    model_outputs=[outputs[i]], # wrap single output to match single-item API
                    w=w, h=h,
                    ignored_tokens=self.ignored_tokens,
                    point_var_ration=self.config['ui_zoom_point_var_ratio'],
                    confidence_threshold=self.config['ui_zoom_confidence_threshold'],
                    min_side=self.config['ui_zoom_min_crop'],
                    scale=self.config['ui_zoom_scale'],
                ))
            except Exception as err:
                logger.error(f"Aguvis Ui Zoom Batch Stage 1 [idx={i}]: {err}")
                results[i] = AgentOutput(raw={"error": str(err)})
                continue
            if ui_zoomer_output.use_voted_point:
                results[i] = AgentOutput(
                    coordinate=(ui_zoomer_output.voted_point[0] / w, ui_zoomer_output.voted_point[1] / h), # voted point normalize by w,h
                    action_type='click',
                    raw={"original_model_outputs": output_texts_list[i], "ui_zoomer_output": ui_zoomer_output.model_dump()}
                )
            else:
                needs_stage2.append((i, ui_zoomer_output, w, h))

        # === Phase 2 ===
        if needs_stage2:
            # rerun on cropped images
            cropped_images = [screenshots[i].copy().crop(ui_zoomer_output.bbox) for i, ui_zoomer_output, _, _ in needs_stage2]
            sampling_params = SamplingParams(temperature=1, max_tokens=self.config['max_new_tokens'])
            vllm_inputs = [{
                "prompt": self._get_model_inputs(cropped_image, tasks[i], self.config['grounding_system_message'])[2],
                "multi_modal_data": {"image": cropped_image},
            } for cropped_image, (i, _, _, _) in zip(cropped_images, needs_stage2)]

            cropped_outputs = self.model.generate(vllm_inputs, [sampling_params] * len(needs_stage2))

            for (i, ui_zoomer_output, w, h), cropped_image, cropped_out in zip(needs_stage2, cropped_outputs, cropped_images):
                wc, hc = cropped_image.size
                cropped_output_texts = [raw_out.text.strip() for raw_out in cropped_out.outputs]
                try:
                    cropped_point = get_points_from([cropped_out], wc, hc)[0] # wrap single output to match single-item API; multiple by wc, hc
                except Exception as err:
                    logger.error(f"Aguvis Ui Zoom Batch Stage 2 [idx={i}]: {err}")
                    results[i] = AgentOutput(raw={"error": str(err)})
                    continue
                x_pred = cropped_point[0] + ui_zoomer_output.bbox[0] # x*wc + x0
                y_pred = cropped_point[1] + ui_zoomer_output.bbox[1] # y*hc + y0
                results[i] = AgentOutput(
                    coordinate=(x_pred / w, y_pred / h), # voted point normalize by w,h
                    action_type='click',
                    raw={"original_model_outputs": output_texts_list[i], "cropped_model_outputs": cropped_output_texts, "ui_zoomer_output": ui_zoomer_output.model_dump()}
                )

        return results

        
        
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
