from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.utils import get_torch_dtype

import re
import torch
from pathlib import Path
from loguru import logger
from PIL.Image import Image, Resampling
from typing import Any
from qwen_vl_utils import smart_resize
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor


class UIAgileAgent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)

        self.config.setdefault("max_new_tokens", 128)        # more headroom for <think> tokens
        self.config.setdefault("model_path", str(MODELS_PATH / "ui-agile-3b"))
        self.config.setdefault("repo_id", "KDEGroup/UI-AGILE-3B")
        self.config.setdefault("dtype", "bfloat16")           # model card says BF16
        self.config.setdefault("device_map", "auto")
        self.config.setdefault("thinking", True)              # UI-AGILE uses "Simple Thinking" reward

        self.dtype = get_torch_dtype(self.config["dtype"])

        self.model_path = Path(self.config["model_path"])
        if not self.model_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                local_dir=str(self.config["model_path"]),
                local_dir_use_symlinks=False,
            )

    def load(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.config["device_map"],
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # Public predict interface
    # ------------------------------------------------------------------

    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        screenshot_processed = self.preprocess(screenshot)
        messages = self._get_grounding_chat_messages(screenshot_processed, task)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=self.config["thinking"],
        )
        inputs = self.processor(
            text=[text], images=[screenshot_processed], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"]
            )
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )
        return self.postprocess_grounding(
            output_text[0].strip(), screenshot_processed.width, screenshot_processed.height
        )

    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        texts, all_images = [], []
        for screenshot, task in inputs:
            screenshot_processed = self.preprocess(screenshot)
            messages = self._get_grounding_chat_messages(screenshot_processed, task)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking=self.config["thinking"],
            )
            texts.append(text)
            all_images.append(screenshot_processed)

        batch_inputs = self.processor(
            text=texts, images=all_images, padding=True, return_tensors="pt"
        )
        batch_inputs = batch_inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **batch_inputs, max_new_tokens=self.config["max_new_tokens"]
            )
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(batch_inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )
        return [
            self.postprocess_grounding(t.strip(), img.width, img.height)
            for t, img in zip(output_texts, all_images)
        ]

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def preprocess(self, screenshot: Image) -> Image:
        """Resize to the processor's expected resolution using smart_resize."""
        image_processor_config = self.processor.image_processor
        resized_height, resized_width = smart_resize(
            screenshot.height,
            screenshot.width,
            factor=image_processor_config.patch_size * image_processor_config.merge_size,
            min_pixels=image_processor_config.size.get("shortest_edge", None),
            max_pixels=image_processor_config.size.get("longest_edge", None),
        )
        return screenshot.resize(
            size=(resized_width, resized_height), resample=Resampling.LANCZOS
        )

    def postprocess_grounding(
        self, raw_output: str, img_width: int, img_height: int
    ) -> AgentOutput:
        """
        UI-AGILE / Qwen2.5-VL outputs absolute pixel coordinates, e.g.:
          <point x="512" y="384">element</point>
          or plain:  (512, 384)
        We normalise to [0, 1] to match the shared AgentOutput contract.
        """
        try:
            # Pattern 1: <point x="..." y="...">...</point>
            point_match = re.search(
                r'<point\s+x=["\']?(\d+)["\']?\s+y=["\']?(\d+)["\']?', raw_output
            )
            if point_match:
                x_abs = int(point_match.group(1))
                y_abs = int(point_match.group(2))
            else:
                # Pattern 2: bare (x, y) tuple anywhere in the string
                tuple_match = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", raw_output)
                if tuple_match:
                    x_abs = int(tuple_match.group(1))
                    y_abs = int(tuple_match.group(2))
                else:
                    raise ValueError(f"No coordinate pattern found in: {raw_output!r}")

            # Normalise absolute pixels → [0, 1]
            x = x_abs / img_width
            y = y_abs / img_height

            return AgentOutput(
                coordinate=(x, y),
                action_type="click",
                raw={"content": raw_output},
            )
        except Exception as err:
            logger.error(
                f"UIAgile: Failed to parse grounding output: raw_output={raw_output!r}\n"
                f"error: {err}"
            )
            return AgentOutput(raw={"content": raw_output})

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _get_grounding_chat_messages(
        self, screenshot: Image, task: str
    ) -> list[dict[str, any]]:
        """
        UI-AGILE is trained on the standard Qwen2.5-VL GUI-grounding prompt:
        a system turn describing the agent role, followed by a user turn
        with the screenshot and the target instruction.
        """
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MSG}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": GROUNDING_MSG.format(task=task)},
                ],
            },
        ]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are a GUI agent. You are given a screenshot of a graphical user interface "
    "and a natural language instruction describing an element to locate. "
    "Your task is to identify and output the click coordinates of the target element."
)

GROUNDING_MSG = (
    "Please find the element described below and output its location as a point "
    "in the format <point x=\"X\" y=\"Y\">element</point>, "
    "where X and Y are pixel coordinates on the screenshot.\n\n"
    "Target: {task}"
)