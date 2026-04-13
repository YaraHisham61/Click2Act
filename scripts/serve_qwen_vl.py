# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-11
# Prompt : Option A serving — FastAPI + bitsandbytes with --quantize {4bit,8bit,none} flag;
#          OpenAI-compatible /v1/chat/completions endpoint for Qwen2.5-VL-7B-Instruct

"""
Serve Qwen2.5-VL-7B-Instruct as a local OpenAI-compatible API.

Usage:
    conda run -n fast_env python -m scripts.serve_qwen_vl --quantize 4bit
    conda run -n fast_env python -m scripts.serve_qwen_vl --quantize none --port 8001

Endpoints:
    GET  /health
    POST /v1/chat/completions

Images can be passed as base64 data URIs or local file paths in content blocks.
"""

import argparse
import base64
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Qwen2.5-VL via FastAPI")
    p.add_argument(
        "--model-path",
        default="./models/qwen2.5-vl-7b",
        help="Path to the model directory (default: ./models/qwen2.5-vl-7b)",
    )
    p.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization level: 4bit (~4 GB VRAM), 8bit (~8 GB), none/fp16 (~14 GB) (default: 4bit)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, quantize: str):
    print(f"[serve_qwen_vl] Loading model from '{model_path}' (quantize={quantize}) ...")

    bnb_config = None
    if quantize == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if quantize == "none" else None,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print("[serve_qwen_vl] Model ready.")
    return model, processor


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(source: str) -> Image.Image:
    """Accept base64 data URI or local file path."""
    if source.startswith("data:image"):
        # data:image/<fmt>;base64,<data>
        b64 = re.split(r"base64,", source, maxsplit=1)[1]
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    return Image.open(Path(source)).convert("RGB")


# ---------------------------------------------------------------------------
# Pydantic schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ContentPart(BaseModel):
    type: str                          # "text" | "image_url"
    text: str | None = None
    image_url: dict | None = None      # {"url": "..."}


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]   # plain string or multimodal list


class ChatRequest(BaseModel):
    model: str = "qwen2.5-vl-7b"
    messages: list[Message]
    max_new_tokens: int = 512
    temperature: float = 1.0
    do_sample: bool = False


# ---------------------------------------------------------------------------
# App factory (model injected at startup)
# ---------------------------------------------------------------------------

def create_app(model, processor) -> FastAPI:
    app = FastAPI(title="Qwen2.5-VL Inference Server")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "qwen2.5-vl-7b"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        # Build qwen-style message list with PIL images extracted
        messages = []
        images: list[Image.Image] = []

        for msg in req.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
                continue

            parts = []
            for part in msg.content:
                if part.type == "text":
                    parts.append({"type": "text", "text": part.text or ""})
                elif part.type == "image_url" and part.image_url:
                    try:
                        img = _load_image(part.image_url["url"])
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Cannot load image: {e}")
                    images.append(img)
                    parts.append({"type": "image"})

            messages.append({"role": msg.role, "content": parts})

        try:
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=images if images else None,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    do_sample=req.do_sample,
                )

            # Strip the prompt tokens from the output
            prompt_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, prompt_len:]
            response_text = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # OpenAI-compatible response envelope
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(generated[0]),
                "total_tokens": prompt_len + len(generated[0]),
            },
        })

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    model, processor = load_model(args.model_path, args.quantize)
    app = create_app(model, processor)
    uvicorn.run(app, host=args.host, port=args.port)
