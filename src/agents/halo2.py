from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.utils import get_torch_dtype

import re
import json
import torch
from pathlib import Path
from loguru import logger
from PIL.Image import Image, Resampling
from pydantic import BaseModel, Field
from typing import Any, Literal, TypeAlias
from qwen_vl_utils import process_vision_info, smart_resize
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor


class ClickCoordinates(BaseModel):
    x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")


class HALO2Agent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)
        
        self.config.setdefault("max_new_tokens", 32)
        self.config.setdefault("model_path", str(MODELS_PATH / "halo2-4b"))
        self.config.setdefault("repo_id", "Hcompany/Holo2-4B")
        self.config.setdefault("dtype", "float16")
        self.config.setdefault("device_map", "auto")
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
        texts, all_images = [], []
        for screenshot, task in inputs:
            screenshot_processed = self.preprocess(screenshot)
            messages = self._get_grounding_chat_messages(screenshot_processed, task)
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

    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        # prepare image input
        screenshot_processed = self.preprocess(screenshot)
        # prepare model inputs
        messages = self._get_grounding_chat_messages(screenshot_processed, task)
        # >>> thinking false because we need grounding only
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=False)
        inputs = self.processor(text=[text], images=[screenshot_processed], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        # generate ids
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.config['max_new_tokens'])
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # decode ids
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        return self.postprocess_grounding(output_text[0].strip())
    
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
            structured_output = ClickCoordinates(**json.loads(raw_output))
            # since output normalized bet 0-1000 and I want to be bet 0-1
            x, y = structured_output.x/1000, structured_output.y/1000
            return AgentOutput(
                coordinate=(x, y),
                action_type="click",
                raw = {"content": raw_output}
            )
        except Exception as err:
            logger.error(f"HALO2: This action not handled to be parsed yet: raw_output={raw_output}\n error {err}")
            return AgentOutput(raw = {"content": raw_output})
    
    def _get_grounding_chat_messages(self, screenshot: Image, task: str, history: str = "None") -> list[dict[str, any]]:
        """Create the prompt structure for navigation task"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": f"{GROUNDING_MSG}\t{task}"},
                ],
            },
        ]
        
# --- Prompts Ref: are taking from Halo2
GROUNDING_MSG = (
        "Localize an element on the GUI image according to the provided target and output a click position.\n"
        f"* You must output a valid JSON following the format: {ClickCoordinates.model_json_schema()}\n"
        "Your target is:\n"
)