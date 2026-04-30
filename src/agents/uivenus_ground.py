from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.utils import get_torch_dtype

import ast
import json
import re
import torch
from typing import Any
from pathlib import Path
from loguru import logger
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    # Fallback for environments where Qwen2.5 VL class is unavailable.
    from transformers import AutoModelForImageTextToText as Qwen2_5_VLForConditionalGeneration, AutoProcessor


class UIVenusGroundAgent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)

        self.config.setdefault("max_new_tokens", 128)
        self.config.setdefault("temperature", 0.0)
        self.config.setdefault("do_sample", False)
        self.config.setdefault("model_path", str(MODELS_PATH / "ui-venus-ground-7b"))
        self.config.setdefault("repo_id", "inclusionAI/UI-Venus-Ground-7B")
        self.config.setdefault("dtype", "bfloat16")
        self.config.setdefault("device_map", "auto")
        self.config.setdefault("attn_implementation", None)
        self.config.setdefault("trust_remote_code", True)
        self.config.setdefault("vision_patch_size", 14)
        self.config.setdefault(
            "grounding_prompt_template",
            "Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2].",
        )

        self.dtype = get_torch_dtype(self.config["dtype"])
        self.model_path = Path(self.config["model_path"])
        self.model: Any = None
        self.processor: Any = None

        if not self.model_path.exists() or not self._has_complete_weights(self.model_path):
            snapshot_download(
                repo_id=self.config["repo_id"],
                local_dir=str(self.model_path),
                local_dir_use_symlinks=False,
            )

    def _has_complete_weights(self, model_path: Path) -> bool:
        index_path = model_path / "model.safetensors.index.json"
        if not index_path.exists():
            return False

        try:
            index_data = json.loads(index_path.read_text())
            weight_map = index_data.get("weight_map", {})
            shard_names = set(weight_map.values())
            if not shard_names:
                return False

            return all((model_path / shard).exists() for shard in shard_names)
        except (OSError, json.JSONDecodeError, TypeError):
            return False

    def load(self):
        load_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.config["device_map"],
            "trust_remote_code": self.config["trust_remote_code"],
        }
        if self.config["attn_implementation"]:
            load_kwargs["attn_implementation"] = self.config["attn_implementation"]

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **load_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.config["trust_remote_code"],
        )

    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        return self.predict_click_batch([(screenshot, task)])[0]

    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        messages_batch: list[list[dict[str, any]]] = []
        texts: list[str] = []

        for screenshot, task in inputs:
            messages = self._get_grounding_messages(screenshot, task)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            messages_batch.append(messages)
            texts.append(text)

        image_inputs, video_inputs = process_vision_info(messages_batch)
        model_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=self.config["do_sample"],
                temperature=self.config["temperature"],
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        image_grid_thw = model_inputs.get("image_grid_thw", None)
        dims = self._resolve_input_dims(image_grid_thw, len(output_texts))

        return [
            self.postprocess_grounding(raw_output=raw_output.strip(), input_dims=dim)
            for raw_output, dim in zip(output_texts, dims)
        ]

    def postprocess_grounding(self, raw_output: str, input_dims: tuple[float, float]) -> AgentOutput:
        input_width, input_height = input_dims

        try:
            x1, y1, x2, y2 = self._extract_box(raw_output)
            x1 = x1 / input_width
            x2 = x2 / input_width
            y1 = y1 / input_height
            y2 = y2 / input_height

            x = max(0.0, min(1.0, (x1 + x2) / 2.0))
            y = max(0.0, min(1.0, (y1 + y2) / 2.0))

            return AgentOutput(
                coordinate=(x, y),
                action_type="click",
                raw={"content": raw_output, "bbox": [x1, y1, x2, y2]},
            )
        except (ValueError, SyntaxError, TypeError) as err:
            logger.error(
                f"UI-VENUS-GROUND: Could not parse output to click point: raw_output={raw_output}\\nerror={err}"
            )
            return AgentOutput(raw={"content": raw_output})

    def _resolve_input_dims(self, image_grid_thw, batch_size: int) -> list[tuple[float, float]]:
        patch = float(self.config["vision_patch_size"])

        if image_grid_thw is None:
            return [(1000.0, 1000.0)] * batch_size

        try:
            grid = image_grid_thw.detach().cpu().tolist()
            dims = [(float(item[2]) * patch, float(item[1]) * patch) for item in grid]

            if len(dims) >= batch_size:
                return dims[:batch_size]
            if len(dims) == 1:
                return dims * batch_size
            return dims + [dims[-1]] * (batch_size - len(dims))
        except (AttributeError, TypeError, ValueError):
            return [(1000.0, 1000.0)] * batch_size

    def _extract_box(self, raw_output: str) -> tuple[float, float, float, float]:
        candidate = raw_output.strip()

        if not candidate.startswith("["):
            match = re.search(r"\[[^\]]+\]", candidate)
            if match:
                candidate = match.group(0)

        box = ast.literal_eval(candidate)
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Output is not a [x1,y1,x2,y2] list.")

        return float(box[0]), float(box[1]), float(box[2]), float(box[3])

    def _get_grounding_messages(self, screenshot: Image, task: str) -> list[dict[str, any]]:
        prompt = self.config["grounding_prompt_template"].format(instruction=task)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
