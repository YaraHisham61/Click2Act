You are a research agent. Your task is to survey the GUI agent landscape as of early 2026.

## Goal
Find models/pipelines/agents for GUI automation that are:
- More recent than OmniParser (Nov 2024), AGUVIS (Dec 2024), or UI-TARS (Jan 2025)
- OR outperform them on standard benchmarks
- OR are significantly smaller (parameter-efficient) while competitive

## Search Targets

### 1. New GUI Agent Models (2025–2026)
For each model found, collect:
- Model name, organization, release date (arXiv / GitHub / HF)
- Key innovation (what makes it different)
- Parameter count if available
- Benchmark scores (OSWorld, MMBench-GUI, ScreenSpot, AndroidWorld, etc.)
- Links: paper (arXiv), code (GitHub), weights (HuggingFace)

Search queries to try:
- site:arxiv.org "GUI agent" 2025
- site:huggingface.co "GUI agent" OR "computer use" OR "UI grounding"
- site:github.com "GUI agent" stars:>100 pushed:>2025-01-01
- "OSWorld" leaderboard top models
- "MMBench-GUI" leaderboard
- "ScreenSpot" benchmark results 2025

### 2. Benchmarks — Popular & Updated on HuggingFace
For each benchmark found, collect:
- Full name and HuggingFace dataset/space link
- arXiv paper link
- What it targets (web, desktop, mobile, cross-platform)
- How it was created (annotation method, source of tasks)
- Metrics used (e.g., task success rate, step success rate, grounding accuracy)
- Top-3 performing models with their scores
- Date of last leaderboard update

Key benchmarks to look up:
- OSWorld — https://huggingface.co/papers/2404.07972
- MMBench-GUI — search HF
- ScreenSpot / ScreenSpot-Pro
- AndroidWorld
- WindowsAgentArena
- GUI-World
- AgentBench
- Mind2Web
- WorkArena
- AssistGUI
- Any NEW benchmark released after Jan 2025

### 3. Output Format
Write a markdown inside `docs/{ai-name}/00-init-search.md` with two sections:

#### Section A — Benchmarks
benchmark_name:
  hf_link:
  paper_link:
  target: (web | desktop | mobile | cross-platform)
  creation_method:
  metrics: []
  top_models:
    - name:
      score:
      date:

#### Section B — Top Models
model_name:
  org:
  release_date:
  params:
  innovation:
  scores:
    benchmark_name: score
  links:
    paper:
    code:
    weights:

## Constraints
- Prioritize results from 2025 onward
- Only include models with published benchmark results (no vaporware)
- For each claim, include a citation URL
- If a leaderboard is found on HuggingFace Spaces, scrape the top-5 entries