from src.constants import MODELS_PATH
from src.agents.base import GUIAgent, AgentOutput
from src.utils import get_torch_dtype

import re
import torch
from pathlib import Path
from loguru import logger
from PIL.Image import Image, Resampling
from qwen_vl_utils import smart_resize
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

class SafeLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=-1e4, neginf=-1e4)
        return scores
class UIAgileAgent(GUIAgent):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)

        self.config.setdefault("max_new_tokens", 64)
        self.config.setdefault("model_path", str(MODELS_PATH / "ui-agile-3b"))
        self.config.setdefault("repo_id", "KDEGroup/UI-AGILE-3B")
        self.config.setdefault("device_map", "auto")
        self.config.setdefault("thinking", False)
        self.config.setdefault("max_pixels", 1003520)
        self.config.setdefault("quantize", False)

        # Qwen2.5-VL overflows with float16 -> promote to bfloat16 unless quantizing
        raw_dtype = self.config.get("dtype", "float16")
        
        raw_dtype = "float16"
        self.config["dtype"] = raw_dtype
        self.dtype = get_torch_dtype(raw_dtype)

        self.model_path = Path(self.config["model_path"])
        if not self.model_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                local_dir=str(self.config["model_path"]),
                local_dir_use_symlinks=False,
            )

    def load(self):
        if self.config["quantize"]:
            # NF4 4-bit: weights stored in 4-bit, computed in bfloat16.
            # Drops weight memory from ~6GB to ~2GB, leaving room for
            # vision encoder activations (~1.4GB) and KV cache on T4.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("UIAgile: loading with NF4 4-bit quantization (BitsAndBytes)")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.config["device_map"],
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=self.config["device_map"],
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"
        self._safe_logits_processor = LogitsProcessorList([SafeLogitsProcessor()])

    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        screenshot_processed = self.preprocess(screenshot)
        messages = self._get_grounding_chat_messages(screenshot_processed, task)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
                messages, tokenize=False, add_generation_prompt=True,
                thinking=self.config["thinking"],
            )
            texts.append(text)
            all_images.append(screenshot_processed)
        try:
            batch_inputs = self.processor(
                text=texts, images=all_images, padding=True, return_tensors="pt"
            )
            batch_inputs = batch_inputs.to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **batch_inputs, max_new_tokens=self.config["max_new_tokens"],
                    logits_processor=self._safe_logits_processor,

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
        except Exception as batch_err:
            # Batch failed (likely NaN/Inf on one image) — fall back to one-by-one
            logger.warning(f"UIAgile: batch failed ({batch_err}), retrying one-by-one...")
            torch.cuda.empty_cache()
            results = []
            for img, text in zip(all_images, texts):
                try:
                    single_input = self.processor(
                        text=[text], images=[img], padding=True, return_tensors="pt"
                    )
                    single_input = single_input.to(self.model.device)
                    with torch.inference_mode():
                        gen_ids = self.model.generate(
                            **single_input, max_new_tokens=self.config["max_new_tokens"]
                        )
                    trimmed = gen_ids[0][single_input.input_ids.shape[1]:]
                    out_text = self.processor.decode(trimmed, skip_special_tokens=True)
                    results.append(
                        self.postprocess_grounding(out_text.strip(), img.width, img.height)
                    )
                except Exception as item_err:
                    logger.error(f"UIAgile: skipping image due to error: {item_err}")
                    torch.cuda.empty_cache()
                    results.append(AgentOutput(raw={"content": ""}))  # empty/skipped
            return results

    def preprocess(self, screenshot: Image) -> Image:
        # smart_resize with hard max_pixels cap to prevent OOM on 16GB GPU
        image_processor_config = self.processor.image_processor
        factor = image_processor_config.patch_size * image_processor_config.merge_size
        max_pixels = self.config.get("max_pixels", 1003520)
        min_pixels = image_processor_config.size.get("shortest_edge", None)
        resized_height, resized_width = smart_resize(
            screenshot.height, screenshot.width,
            factor=factor, min_pixels=min_pixels, max_pixels=max_pixels,
        )
        return screenshot.resize(
            size=(resized_width, resized_height), resample=Resampling.LANCZOS
        )

    def postprocess_grounding(
        self, raw_output: str, img_width: int, img_height: int
    ) -> AgentOutput:
        try:
            point_match = re.search(
                r'<point\s+x=["\']?(\d+)["\']?\s+y=["\']?(\d+)["\']?', raw_output
            )
            if point_match:
                x_abs = int(point_match.group(1))
                y_abs = int(point_match.group(2))
            else:
                tuple_match = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", raw_output)
                if tuple_match:
                    x_abs = int(tuple_match.group(1))
                    y_abs = int(tuple_match.group(2))
                else:
                    raise ValueError(f"No coordinate pattern found in: {raw_output!r}")
            return AgentOutput(
                coordinate=(x_abs / img_width, y_abs / img_height),
                action_type="click",
                raw={"content": raw_output},
            )
        except Exception as err:
            logger.error(f"UIAgile: parse failed: {raw_output!r} | {err}")
            return AgentOutput(raw={"content": raw_output})

    def _get_grounding_chat_messages(
        self, screenshot: Image, task: str
    ) -> list[dict[str, any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": screenshot},
                {"type": "text", "text": GROUNDING_MSG.format(task=task)},
            ]},
        ]


SYSTEM_MSG = (
    "You are a GUI agent. You are given a screenshot of a graphical user interface "
    "and a natural language instruction describing an element to locate. "
    "Your task is to identify and output the click coordinates of the target element."
)

GROUNDING_MSG = (
    'Output only a point tag for the target element, nothing else.\n'
    'Format: <point x="X" y="Y">t</point>\n\n'
    'Target: {task}'
)

