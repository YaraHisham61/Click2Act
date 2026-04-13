<!-- AI-GENERATED
     Model   : Claude Sonnet 4.6
     Date    : 2026-04-11
     Prompt  : "what's best vision models with budget friendly, explore also local models, also if they have free access for first time. I already have gemini api (is there's good vision model they have). write your results in markdown files in docs/claude"
-->

# Budget-Friendly Vision Models Guide (April 2026)

A practical survey of vision-language models (VLMs) organized by cost tier — cloud APIs, Gemini (already available), and local/self-hosted.

---

## 1. Cloud API Models — Cheapest First

| Model | Provider | Input ($/1M tok) | Output ($/1M tok) | Notes |
|---|---|---|---|---|
| **Qwen2.5-VL-7B** | SiliconFlow / Together | ~$0.05 | ~$0.05 | Best price/perf ratio; strong OCR & doc VQA |
| **Gemini 2.5 Flash-Lite** | Google | ~$0.10 | ~$0.40 | Free tier available; see §2 |
| **Gemini 2.5 Flash** | Google | $0.30 | $2.50 | Free tier available; see §2 |
| **DeepSeek-VL2** | DeepSeek | ~$0.14 | ~$0.28 | Strong reasoning + vision |
| **LLaMA 3.2 Vision 11B** | Groq / Together | ~$0.18 | ~$0.18 | Open-weight, fast inference |
| **Claude Haiku 4.5** | Anthropic | $1.00 | $5.00 | Fast, reliable; vision support |
| **Gemini 2.5 Pro** | Google | ~$1.25 | ~$10.00 | Top-tier; NOT on free tier after April 2026 |

> [Uncertain] Exact pricing for SiliconFlow/Together may vary; verify at provider dashboard before large-scale runs.

---

## 2. Gemini API — What You Already Have

Since you have a Gemini API key, these models are immediately usable.

### Free Tier (as of April 2026)

After April 1, 2026 Google restricted Pro models behind a paywall. Free tier now covers **Flash models only**.

| Model | Free RPM | Free TPM | Free Req/Day | Vision? |
|---|---|---|---|---|
| `gemini-2.5-flash` | 15 | 250,000 | 1,000 | Yes (images, video) |
| `gemini-2.5-flash-lite` | 30 | 250,000 | 1,500 | Yes |
| `gemini-2.0-flash` | 15 | 1,000,000 | 1,500 | Yes |

**Important:** Free-tier data may be used by Google to improve their models. Switch to paid tier if working with sensitive screenshots/data.

### Image tokenization

Each image ≈ **258 tokens** regardless of resolution (higher-res images may tokenize slightly higher). At $0.30/1M tokens (Flash), that is ~$0.0001 per image — extremely cheap.

### Recommended model for this project

**`gemini-2.5-flash`** — best balance of capability and cost. Strong at GUI screenshot understanding, OCR, and visual reasoning. The free tier covers early-stage experiments easily (1,000 screenshots/day free).

---

## 3. Local / Self-Hosted Models

Zero API cost after hardware; full data privacy. Useful for bulk inference or offline runs.

### Tier A — Requires VRAM ≥ 8 GB

| Model | Params | VRAM | Ollama Tag | MMMU | DocVQA | GUI tasks |
|---|---|---|---|---|---|---|
| **Qwen2.5-VL 7B** | 7B | ~6 GB | `qwen2.5vl:7b` | 58.6 | 95.7 | Strong (visual agent, no fine-tune) |
| **Qwen3-VL 8B** | 8B | ~7 GB | `qwen3-vl:8b` | Better than 2.5-VL | — | Released late 2025; best local for OCR+charts |
| **Gemma 3 4B** | 4B | ~4 GB | `gemma3:4b` | — | — | Lightweight; 128k context |
| **LLaMA 3.2 Vision 11B** | 11B | ~10 GB | `llama3.2-vision:11b` | 60.3 | — | Strong OCR + document VQA |

### Tier B — Requires VRAM ≥ 16–24 GB

| Model | Params | Active Params | Notes |
|---|---|---|---|
| **InternVL 3.5-20B-A4B** | 20B (MoE) | ~4B active | Efficient MoE; competes with much larger models |
| **Qwen2.5-VL 72B** | 72B | 72B | Top open-source VLM; needs A100/H100 |
| **Gemma 3 27B** | 27B | 27B | MMMU 64.9; best mid-size consumer model |
| **LLaMA 3.2 Vision 90B** | 90B | 90B | Near-frontier quality; 2× A100 minimum |

### Tier C — CPU / Edge (< 8 GB RAM)

| Model | Params | Use case |
|---|---|---|
| **SmolVLM2 2.2B** | 2.2B | Fastest local; limited reasoning |
| **SmolVLM2 500M** | 500M | Embedded / resource-constrained |

### Running locally with Ollama

```bash
# Install Ollama (if not already)
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run Qwen2.5-VL 7B
ollama run qwen2.5vl:7b

# Pull Qwen3-VL 8B (newer)
ollama run qwen3-vl:8b
```

---

## 4. Recommendation Matrix

| Scenario | Recommended model | Why |
|---|---|---|
| **Rapid prototyping, free** | `gemini-2.5-flash` (free tier) | 1k req/day free, strong vision, already available |
| **Bulk inference on budget** | `Qwen2.5-VL-7B` via SiliconFlow API | $0.05/1M tokens — 6× cheaper than Flash |
| **Local, 8 GB GPU** | `qwen3-vl:8b` via Ollama | Best quality at that VRAM tier |
| **Local, 16+ GB GPU** | `InternVL 3.5-20B-A4B` | MoE efficiency; only 4B active params |
| **High accuracy, paid** | `gemini-2.5-pro` or `GPT-4o` | When quality matters most |
| **GUI / screen agent tasks** | `Qwen2.5-VL 7B` or `Qwen3-VL` | Explicitly designed for visual agents |

---

## 5. Relevance to This Project (Click2Act)

For benchmarking on **OSWorld** and **MMBench-GUI**:

- `gemini-2.5-flash` (free tier) is the fastest path to a first baseline — no cost, no setup, already authenticated.
- `Qwen2.5-VL 7B` local is the best open-weight baseline for screen understanding; directly comparable to OmniParser's backbone.
- `InternVL 3.5-20B-A4B` is worth testing if a 16+ GB GPU is available — competes with GPT-4V-level models at zero API cost.

---

## Sources

- [Gemini API Pricing (Updated April 2026)](https://www.tldl.io/resources/google-gemini-api-pricing)
- [Gemini Developer API Pricing — Official](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini Free Tier Limits 2026](https://blog.laozhang.ai/en/posts/gemini-api-free-tier)
- [Cheapest LLM Models 2026 — SiliconFlow](https://www.siliconflow.com/articles/en/the-cheapest-LLM-models)
- [Best Local Vision-Language Models — Roboflow](https://blog.roboflow.com/local-vision-language-models/)
- [Best Vision Models Locally — InsiderLLM](https://insiderllm.com/guides/vision-models-locally/)
- [Open-Source VLMs 2026 — BentoML](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [Top VLMs 2026 — DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Qwen2.5-VL on Ollama](https://ollama.com/library/qwen2.5vl)
- [LLM API Pricing Comparison Apr 2026 — CostGoat](https://costgoat.com/compare/llm-api)
