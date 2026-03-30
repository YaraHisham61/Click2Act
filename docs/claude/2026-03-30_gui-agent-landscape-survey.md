<!-- AI-GENERATED
     Model   : Claude Sonnet 4.6
     Date    : 2026-03-30
     Prompt  : Survey the GUI agent landscape as of early 2026. Find models/pipelines/agents that are
               more recent than OmniParser (Nov 2024), AGUVIS (Dec 2024), or UI-TARS (Jan 2025),
               OR outperform them on standard benchmarks, OR are significantly smaller while competitive.
               Collect model metadata (org, date, params, innovation, scores, links) and benchmark
               metadata (platform, creation method, metrics, top models). Output two YAML-style sections.
    
      Full Prompt Location: prompts/2026-03-30_init-search
-->

# GUI Agent Landscape Survey — Early 2026

Compiled: 2026-03-30. Covers models and benchmarks that post-date or outperform the project's three
baseline systems (OmniParser Nov 2024, AGUVIS Dec 2024, UI-TARS Jan 2025).

---

## Section A — Benchmarks

---

### OSWorld

```yaml
benchmark_name: OSWorld
  full_name: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
  hf_link: https://huggingface.co/papers/2404.07972
  paper_link: https://arxiv.org/abs/2404.07972
  site: https://os-world.github.io/
  target: desktop (Ubuntu, Windows, macOS)
  creation_method: >
    369 real-world tasks derived from human computer-use scenarios. Each task has
    an automated execution-based evaluation script and a defined initial state.
    Tasks span Chrome, LibreOffice, VS Code, Thunderbird, and cross-app workflows.
  metrics:
    - task_success_rate (primary; execution-based, not LLM-judged)
  human_performance: ~72%
  notes: >
    OSWorld-Verified (launched 2025-07-28) is an in-place upgrade with fixed
    tasks, standardized environments, and AWS support. Official results are
    team-verified; community results are self-reported.
  top_models_osworld_verified:
    - name: CoAct-1
      score: "60.76%"
      date: 2025-08
      citation: https://arxiv.org/abs/2508.03923
    - name: "Agent S2.5 w/ o3"
      score: "56.0%"
      date: 2025-07
      citation: https://xlang.ai/blog/osworld-verified
    - name: "GTA1 w/ o3"
      score: "53.1%"
      date: 2025-07
      citation: https://xlang.ai/blog/osworld-verified
    - name: UI-TARS-2
      score: "47.5%"
      date: 2025-09
      citation: https://arxiv.org/abs/2509.02544
    - name: "OpenCUA-72B"
      score: "45.0%"
      date: 2025-08
      citation: https://arxiv.org/abs/2508.09123
  leaderboard_last_updated: 2025-07-28 (verified track)
```

---

### OSWorld-Verified

```yaml
benchmark_name: OSWorld-Verified
  full_name: OSWorld-Verified (in-place upgrade of OSWorld)
  site: https://os-world.github.io/
  paper_link: https://xlang.ai/blog/osworld-verified
  target: desktop (Ubuntu, Windows, macOS)
  creation_method: >
    Same task set as OSWorld (369 tasks) but with community-reported bugs fixed,
    standardized Docker/AWS environments, and official re-evaluation of all
    submitted models by the OSWorld team.
  metrics:
    - task_success_rate
  notes: >
    Separate self-reported and team-verified tracks.
    llm-stats.com tracks 7 verified models as of early 2026.
  top_models:
    - name: GUI-Owl-1.5 + Mobile-Agent-v3
      score: "46.6% (OSWorld-Verified)"
      date: 2025
    - name: OpenCUA-72B
      score: "45.0%"
      date: 2025-08
    - name: Claude 4 Sonnet (Anthropic)
      score: "43.9%"
      date: 2025
    - name: GUI-Owl-7B + Mobile-Agent-v3
      score: "37.7%"
      date: 2025
    - name: OpenCUA-32B
      score: "34.8%"
      date: 2025-08
  leaderboard_last_updated: 2025-07 (verified)
```

---

### ScreenSpot / ScreenSpot-V2

```yaml
benchmark_name: ScreenSpot / ScreenSpot-V2
  full_name: "ScreenSpot: GUI Grounding Benchmark"
  hf_link: https://huggingface.co/datasets/rootsautomation/ScreenSpot
  paper_link: https://arxiv.org/abs/2401.10935  # original CogAgent/SeeClick paper introduced it
  target: cross-platform (web, desktop, mobile screenshots)
  creation_method: >
    Grounding benchmark: screenshot + natural-language instruction → bounding box.
    ScreenSpot-V2 extends coverage and resolution diversity.
  metrics:
    - grounding_accuracy (click-in-bbox)
  top_models:
    - name: Aria-UI
      score: "82.4%"
      date: 2024-12
      citation: https://arxiv.org/abs/2412.16256
    - name: UI-TARS-72B
      score: "82.8%"
      date: 2025-01
      citation: https://arxiv.org/abs/2501.12326
    - name: GTA1-7B
      score: "92.4% (ScreenSpot-V2)"
      date: 2025-07
      citation: https://arxiv.org/abs/2507.05791
  leaderboard_last_updated: 2025 (ongoing)
```

---

### ScreenSpot-Pro

```yaml
benchmark_name: ScreenSpot-Pro
  full_name: "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use"
  hf_link: https://huggingface.co/blog/Ziyang/screenspot-pro
  paper_link: https://arxiv.org/abs/2504.07981
  site: https://gui-agent.github.io/grounding-leaderboard/
  target: desktop (Windows, macOS, Linux) — professional applications
  creation_method: >
    1,581 unique screenshot–instruction pairs. Each annotated by domain experts
    (≥5 years' experience) across 23 professional applications in 5 domains
    (development, creative, CAD & engineering, scientific, office productivity).
    Average target size: 0.07% of screen area.
  metrics:
    - grounding_accuracy (micro-average, greedy decoding)
  baseline_context: >
    Generalist MLLMs (Qwen2-VL-7B, GPT-4o) score below 2%.
    Specialized 7B models baseline at 16–19%.
  top_models:
    - name: "MAI-UI (Alibaba Tongyi)"
      score: "73.5%"
      date: 2025-12
      citation: https://arxiv.org/abs/2512.22047
    - name: "ScreenSeekeR (cascaded visual search on OS-Atlas-7B)"
      score: "48.1%"
      date: 2025
      citation: https://arxiv.org/abs/2504.07981
    - name: "GTA1-7B"
      score: "50.1%"
      date: 2025-07
      citation: https://arxiv.org/abs/2507.05791
    - name: GUI-Actor-7B (Qwen2.5-VL backbone)
      score: "44.6%"
      date: 2025-06
      citation: https://arxiv.org/abs/2506.03143
    - name: OS-Atlas-7B
      score: "18.9% (baseline)"
      date: 2024-10
      citation: https://arxiv.org/abs/2410.23218
  leaderboard_last_updated: 2025 (ongoing; submit via gui-agent.github.io)
  date_released: 2025-04
```

---

### AndroidWorld

```yaml
benchmark_name: AndroidWorld
  full_name: "AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents"
  paper_link: https://arxiv.org/abs/2405.14573
  target: mobile (Android)
  creation_method: >
    116 core end-to-end tasks across 20 real Android applications
    (Contacts, Camera, Markor, Chrome, Calendar, Files, etc.).
    Tasks are parameterized for randomized instantiation.
  metrics:
    - task_success_rate
  human_performance: ~80%
  top_models:
    - name: "MAI-UI-235B-A22B (Alibaba)"
      score: "76.7%"
      date: 2025-12
      citation: https://arxiv.org/abs/2512.22047
    - name: "MobileRL-9B"
      score: "80.2%"
      date: 2025-09
      citation: https://arxiv.org/abs/2509.18119
    - name: "Mobile-Agent-v3 (GUI-Owl-7B + framework)"
      score: "73.3%"
      date: 2025-08
      citation: https://arxiv.org/abs/2508.15144
    - name: "UI-TARS-2"
      score: "73.3%"
      date: 2025-09
      citation: https://arxiv.org/abs/2509.02544
    - name: "GUI-Owl-1.5"
      score: "71.6%"
      date: 2025
  leaderboard_last_updated: 2025 (ongoing)
  notes: >
    MobileRL-7B outperforms UI-TARS-1.5-72B (+16%) despite being
    a 7B model, demonstrating the power of online RL for mobile tasks.
    Benchmark may be approaching saturation (some systems >80%).
```

---

### WindowsAgentArena

```yaml
benchmark_name: WindowsAgentArena
  full_name: "Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale"
  site: https://microsoft.github.io/WindowsAgentArena/
  paper_link: https://openreview.net/forum?id=t9JUTS9ADL
  target: desktop (Windows OS)
  creation_method: >
    Tasks performed freely within a real Windows OS across diverse applications
    and web browsers. Reproducible via Azure VMs.
  metrics:
    - task_success_rate
  human_performance: ~74.5%
  top_models:
    - name: CoAct-1
      score: "52.5%"
      date: 2025-08
      citation: https://arxiv.org/abs/2508.03923
    - name: UI-TARS-2
      score: "50.6%"
      date: 2025-09
      citation: https://arxiv.org/abs/2509.02544
    - name: "Agent S2"
      score: "29.8%"
      date: 2025-04
      citation: https://arxiv.org/abs/2504.00906
    - name: Navi (original baseline)
      score: "19.5%"
      date: 2024
  leaderboard_last_updated: 2025 (self-reported)
```

---

### MMBench-GUI

```yaml
benchmark_name: MMBench-GUI
  full_name: "MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agents"
  hf_link: https://huggingface.co/datasets/OpenGVLab/MMBench-GUI
  paper_link: https://arxiv.org/abs/2507.19478
  github: https://github.com/open-compass/MMBench-GUI
  target: cross-platform (Windows, Linux, macOS, iOS, Android, Web)
  creation_method: >
    Hierarchical evaluation across 4 levels: GUI Content Understanding,
    GUI Element Grounding, GUI Task Automation, GUI Task Collaboration.
    Expert-annotated tasks across 6 platforms.
  metrics:
    - hierarchical_success_rate (per level)
    - step_efficiency (models show excessive redundant steps)
  top_models: >
    Public leaderboard listed as "coming soon" in paper (Jul 2025).
    Models evaluated internally include GPT-4o, Claude-3.7, Qwen-Max-VL,
    UI-TARS, InternVL, AGUVIS, ShowUI, OS-Atlas.
    Finding: all models suffer from redundant steps even when completing tasks.
  date_released: 2025-07
  leaderboard_last_updated: TBD
```

---

### Online-Mind2Web

```yaml
benchmark_name: Online-Mind2Web
  full_name: "Online-Mind2Web: An Illusion of Progress? Assessing the Current State of Web Agents"
  github: https://github.com/OSU-NLP-Group/Online-Mind2Web
  paper_link: https://arxiv.org/abs/2504.01382
  target: web (live websites)
  creation_method: >
    300 diverse tasks across 136 real websites. Unlike original Mind2Web
    (cached static pages), this runs on live sites (cookies, pop-ups,
    dynamic layouts). Accepted COLM 2025.
  metrics:
    - task_success_rate
  top_models:
    - name: "OpenAI Operator"
      score: "61%"
      date: 2025
    - name: "Claude Computer Use 3.7"
      score: "competitive (top-2)"
      date: 2025
  notes: >
    Most prior agents underperform simple SeeAct (early 2024) on this benchmark,
    exposing overfitting to static cached benchmarks.
  date_released: 2025-04
```

---

### OSUniverse

```yaml
benchmark_name: OSUniverse
  full_name: "OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents"
  site: https://agentsea.github.io/osuniverse/
  paper_link: https://arxiv.org/abs/2505.03570
  target: desktop (Docker containers, OS-agnostic)
  creation_method: >
    Tasks of increasing complexity (basic precision clicking → multistep
    multi-application). Validated using Gemini 2.0/2.5 with long context.
    Calibrated so SOTA agents score <50% while average knowledge workers
    score 100%.
  metrics:
    - task_success_rate (by complexity tier)
  date_released: 2025-05
  notes: Specifically designed to avoid saturation; extensible via Docker.
```

---

### WorldGUI

```yaml
benchmark_name: WorldGUI
  full_name: "WorldGUI: An Interactive Benchmark for Desktop GUI Automation from Any Starting Point"
  paper_link: https://arxiv.org/abs/2502.08047
  github: https://github.com/showlab/GUI-Thinker
  target: desktop (Windows + web apps)
  creation_method: >
    Tasks across 10 widely-used desktop/web apps (PowerPoint, VSCode, Acrobat, etc.).
    Each task instantiated with diverse initial states via controlled pre-actions,
    enabling evaluation of planning robustness and recoverability.
  metrics:
    - task_success_rate
  top_models:
    - name: WorldGUI-Agent
      score: "31.2% (WindowsAgentArena); +12.4% over Claude-3.5 CU on WorldGUI"
      date: 2025-02
  date_released: 2025-02
```

---

### MobileWorld

```yaml
benchmark_name: MobileWorld
  full_name: "MobileWorld: Benchmarking Autonomous Mobile Agents in Agent-User Interactive and MCP-Augmented Environments"
  paper_link: https://arxiv.org/abs/2512.19432
  target: mobile (Android)
  creation_method: >
    Introduces agent-user interaction tasks and MCP (Model-Context-Protocol)
    tool-augmented tasks, extending beyond AndroidWorld's pure GUI scope.
  metrics:
    - task_success_rate
  top_models:
    - name: MAI-UI-235B-A22B
      score: "41.7%"
      date: 2025-12
      citation: https://arxiv.org/abs/2512.22047
  date_released: 2025-01
```

---

### ScreenSpot-Pro / GUI Grounding Leaderboard (Summary)

Combined grounding leaderboard at https://gui-agent.github.io/grounding-leaderboard/ tracks both
ScreenSpot-V2 and ScreenSpot-Pro. Key methods achieving top scores on ScreenSpot-Pro:

| Model / Method | ScreenSpot-Pro | Notes |
|---|---|---|
| MAI-UI | 73.5% | Alibaba Tongyi, Dec 2025 |
| GTA1-7B | 50.1% | Salesforce, Jul 2025 |
| ScreenSeekeR | 48.1% | Inference-only search on OS-Atlas-7B |
| GUI-Actor-7B | 44.6% | Microsoft, NeurIPS 2025 Spotlight |
| OS-Atlas-7B | 18.9% | Baseline (Oct 2024) |
| UI-TARS-2B | 27.7% | ByteDance |
| AriaUI | 11.3% | Aria team |

---

## Section B — Top Models (Post-Jan 2025)

---

### UI-TARS-2

```yaml
model_name: UI-TARS-2
  org: ByteDance Seed
  release_date: 2025-09
  params: not publicly disclosed (successor to UI-TARS-1.5)
  innovation: >
    Multi-turn reinforcement learning with a data flywheel for scalable data
    generation. Hybrid GUI environment integrating file system and terminal.
    Unified sandbox for large-scale rollouts. "All-In-One" agent covering GUI,
    code, tool use, and games.
  scores:
    OSWorld: "47.5%"
    WindowsAgentArena: "50.6%"
    AndroidWorld: "73.3%"
    Online-Mind2Web: "88.2%"
    TerminalBench: "45.3%"
    SWE-Bench: "68.7%"
  links:
    paper: https://arxiv.org/abs/2509.02544
    code: https://github.com/bytedance/UI-TARS
    weights: https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B  # 1.5 public; 2 not yet public
  notes: >
    Surpasses UI-TARS-1.5 on all benchmarks. Positions as a multi-domain
    agent (GUI + code + games), not just a GUI specialist.
```

---

### CoAct-1

```yaml
model_name: CoAct-1
  org: AG2 / Microsoft (Chi Wang et al.)
  release_date: 2025-08
  params: framework (uses frontier LLM backends)
  innovation: >
    Multi-agent architecture combining GUI Operator with a Programmer agent that
    writes and executes Python/Bash scripts. Orchestrator dynamically delegates
    subtasks. Bypasses slow GUI sequences for file/data tasks via direct code.
    Achieves mean 10.15 steps/task vs 15+ for GUI-only agents.
  scores:
    OSWorld: "60.76% (SOTA as of Aug 2025)"
    WindowsAgentArena: "52.5%"
  links:
    paper: https://arxiv.org/abs/2508.03923
    hf: https://huggingface.co/papers/2508.03923
  notes: >
    OSWorld SOTA at time of publication. Strong argument for hybrid GUI+code
    architectures. Efficiency gain (fewer steps) is significant.
```

---

### GTA1 (GUI Test-time Scaling Agent)

```yaml
model_name: GTA1
  org: Salesforce AI Research
  release_date: 2025-07
  params: 7B (GTA1-7B-2507)
  innovation: >
    Test-time scaling for planning: samples multiple action proposals and
    evaluates with a judge model (trades compute for quality). RL-based
    grounding module trained with reward signals for successful clicks.
    Decoupled planning and grounding allow independent scaling.
  scores:
    OSWorld: "45.2% (task success, SOTA among 7B at release)"
    ScreenSpot_Pro: "50.1%"
    ScreenSpot_V2: "92.4%"
    OSWorld_G_grounding: "67.7%"
  links:
    paper: https://arxiv.org/abs/2507.05791
    code: https://github.com/Yan98/GTA1
    weights: https://huggingface.co/Salesforce/GTA1-7B-2507
  notes: >
    Published ICLR 2026. Key result: a 7B model reaches 45%+ on OSWorld
    through test-time compute scaling, competitive with much larger models.
    Strong evidence that inference-time scaling transfers to GUI tasks.
```

---

### OpenCUA (OpenCUA-72B / OpenCUA-32B)

```yaml
model_name: OpenCUA
  org: XLANG AI (OSWorld team)
  release_date: 2025-08-13
  params: "7B, 32B, 72B variants"
  innovation: >
    First large-scale open-source computer-use dataset (AgentNet): 3 OS,
    200+ apps/websites. Includes AgentNetTool (annotation infrastructure),
    AgentNetBench (offline evaluator), and end-to-end foundation models.
    Open-weight alternatives to proprietary CUAs.
  scores:
    OSWorld_Verified:
      OpenCUA-72B: "45.0% (SOTA open-source at release)"
      OpenCUA-32B: "34.8%"
      OpenCUA-7B: "lower"
  links:
    paper: https://arxiv.org/abs/2508.09123
    site: https://opencua.xlang.ai/
    code: https://github.com/xlang-ai/OpenCUA
  notes: >
    Released by the same team that built OSWorld. Provides a fully open
    stack (data + model + eval). OpenCUA-72B surpassed GPT-4o CUA at release.
```

---

### GUI-Owl / GUI-Owl-1.5

```yaml
model_name: GUI-Owl / GUI-Owl-1.5
  org: (paper authors; framework integrated with Mobile-Agent-v3)
  release_date: 2025
  params: 7B
  innovation: >
    Native end-to-end multimodal agent functioning both as a standalone GUI
    agent and as a grounding/reasoning backbone for multi-agent frameworks
    (e.g., Mobile-Agent-v3). SOTA across 10 GUI benchmarks at 7B scale.
  scores:
    AndroidWorld_standalone: "66.4% (GUI-Owl-7B)"
    OSWorld_Verified_standalone: "29.4% (GUI-Owl-7B)"
    AndroidWorld_with_MobileAgentV3: "73.3%"
    OSWorld_Verified_with_MobileAgentV3: "37.7%"
    AndroidWorld_GUI_Owl_1_5: "71.6%"
    OSWorld_Verified_GUI_Owl_1_5: "56.5%"
  links:
    paper: https://arxiv.org/abs/2508.15144  # Mobile-Agent-v3 paper
  notes: >
    GUI-Owl-1.5 claims to outperform UI-TARS-2, Claude-4, and Gemini-2.5-Pro
    on OSWorld-Verified (56.5%) and AndroidWorld (71.6%).
    Strong parameter-efficiency story at 7B scale.
```

---

### AutoGLM-OS / ComputerRL

```yaml
model_name: AutoGLM-OS-9B / AutoGLM-OS-14B
  org: Zhipu AI (ChatGLM team)
  release_date: 2025-08
  params: "9B (GLM-4-9B base) and 14B (Qwen2.5-14B base)"
  innovation: >
    ComputerRL: end-to-end online reinforcement learning for computer use.
    Entropulse training strategy alternates RL with SFT to prevent entropy
    collapse during long training runs. Self-evolving online curriculum RL.
  scores:
    OSWorld: "48.1% (AutoGLM-OS-9B, SOTA at time of ComputerRL paper)"
  links:
    paper_autoglm: https://arxiv.org/abs/2411.00820
    paper_computerrl: https://arxiv.org/abs/2508.14040
    code: https://github.com/zai-org/Open-AutoGLM
  notes: >
    Demonstrates that 9B/14B models trained with online RL can match or
    exceed larger supervised models. Entropulse is a broadly applicable
    training technique for long-horizon agentic RL.
```

---

### Agent S2

```yaml
model_name: Agent S2
  org: Simular Research
  release_date: 2025-04-01
  params: framework (LLM-agnostic; tested with Claude-3.5-Sonnet)
  innovation: >
    Compositional Generalist-Specialist framework. Mixture-of-Grounding (MoG)
    technique for precise GUI localization. Proactive Hierarchical Planning
    for dynamic plan refinement. Modular design separates grounding from
    high-level reasoning.
  scores:
    OSWorld_15step: "27.0% (↑18.9% over Agent S1)"
    OSWorld_50step: "34.5% (↑32.7%)"
    WindowsAgentArena: "29.8% (↑52.8%)"
    AndroidWorld: "54.3% (↑16.5%)"
  links:
    paper: https://arxiv.org/abs/2504.00906
    hf: https://huggingface.co/papers/2504.00906
    code: https://github.com/simular-ai/Agent-S
  notes: >
    Outperforms Claude Computer Use with Claude-3.7-Sonnet by 58.1% (15-step)
    despite using an older backbone (Claude-3.5-Sonnet new). Strong argument
    for hierarchical/modular frameworks over monolithic models.
```

---

### MobileRL

```yaml
model_name: MobileRL (MobileRL-7B / MobileRL-9B)
  org: (paper authors; arxiv 2509.18119)
  release_date: 2025-09
  params: "7B, 9B"
  innovation: >
    Online agentic RL for mobile GUI agents. Difficulty-Adaptive GRPO
    (ADAGRPO): difficulty-adaptive positive replay, failure curriculum
    filtering, and shortest-path reward adjustment for multi-turn tasks.
  scores:
    AndroidWorld_MobileRL-9B: "80.2% (SOTA on AndroidWorld)"
    AndroidLab_MobileRL-9B: "53.6%"
    AndroidWorld_MobileRL-7B: "outperforms UI-TARS-1.5-72B by +16%"
  links:
    paper: https://arxiv.org/abs/2509.18119
  notes: >
    MobileRL-7B beats a 72B model by 16 points on AndroidWorld — a major
    efficiency result. Strongest evidence for RL over SFT on mobile tasks.
    ADAGRPO is a generalizable RL technique for variable-length agentic tasks.
```

---

### Fara-7B

```yaml
model_name: Fara-7B
  org: Microsoft Research
  release_date: 2025-11-24
  params: 7B
  innovation: >
    Microsoft's first agentic SLM for computer use. "Pixel-in, action-out"
    formulation: raw screenshots → intermediate reasoning → atomic actions.
    Trained with FaraGen (synthetic data generation): 145K trajectories,
    1M steps across 70K unique domains. Filters trajectories with multiple
    verifiers. Averages only ~16 steps/task vs ~41 for comparable models.
  scores:
    WebVoyager: "SOTA in class"
    Online-Mind2Web: "SOTA in class"
    WebTailBench: "SOTA in class"
  links:
    paper: https://arxiv.org/abs/2511.19663
    code: https://github.com/microsoft/fara
    weights: https://huggingface.co/microsoft/Fara-7B
    blog: https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/
  notes: >
    Primarily a web-task agent (not yet evaluated on OSWorld/AndroidWorld).
    Step efficiency (16 vs 41 steps) is a critical practical advantage.
    Open-weight on HuggingFace and Azure AI Foundry.
```

---

### MAI-UI

```yaml
model_name: MAI-UI
  org: Alibaba Tongyi Lab
  release_date: 2025-12
  params: "2B, 8B, 32B, 235B-A22B (MoE)"
  innovation: >
    Self-evolving data pipeline including user interaction and MCP tool calls.
    Device-cloud collaboration system (on-device small model + cloud large model).
    Online RL with optimized parallel environments (32→512 envs: +5.2 pts)
    and step budget scaling (15→50 steps: +4.3 pts). Native agent-user
    interaction and multi-modal tool integration.
  scores:
    AndroidWorld: "76.7% (SOTA at release, Dec 2025)"
    ScreenSpot_Pro: "73.5% (SOTA at release)"
    MMBench_GUI_L2: "91.3%"
    OSWorld_G: "70.9%"
    UI_Vision: "49.2%"
    MobileWorld: "41.7%"
  links:
    paper: https://arxiv.org/abs/2512.22047
    code: https://github.com/Tongyi-MAI/MAI-UI
  notes: >
    Surpasses Gemini-2.5-Pro, Seed1.8, and UI-TARS-2 on AndroidWorld and
    ScreenSpot-Pro. The 235B-A22B MoE variant holds most SOTA claims.
    Device-cloud collaboration reduces cloud calls by 40% and improves
    on-device performance by 33%.
```

---

### GUI-Actor

```yaml
model_name: GUI-Actor
  org: Microsoft Research
  release_date: 2025-06
  params: "3B, 7B (with Qwen2-VL or Qwen2.5-VL backbone)"
  innovation: >
    Coordinate-free visual grounding: replaces text-token coordinate output
    with an attention-based action head that identifies regions without
    generating (x, y) coordinates. Avoids weak spatial-semantic alignment
    and granularity mismatch. Can generate multiple candidate regions in
    one forward pass. NeurIPS 2025 Spotlight.
  scores:
    ScreenSpot_Pro_7B: "44.6% (Qwen2.5-VL backbone)"
    ScreenSpot_Pro_3B: "42.2%"
    improvement_over_UI_TARS: "+9.0 pts (2B), +5.0 pts (7B)"
  links:
    paper: https://arxiv.org/abs/2506.03143
    code: https://github.com/microsoft/GUI-Actor
    site: https://microsoft.github.io/GUI-Actor/
  notes: >
    Coordinate-free paradigm is architecturally distinct from all prior work.
    NeurIPS 2025 Spotlight. Strong on ScreenSpot-Pro without test-time search.
```

---

### Mobile-Agent-v3 / Mobile-Agent-v3.5

```yaml
model_name: Mobile-Agent-v3 / v3.5
  org: X-PLUG (Alibaba)
  release_date: "v3: 2025-08; v3.5: 2026-02"
  params: multi-agent framework (uses GUI-Owl-7B as base agent)
  innovation: >
    Cross-platform multi-agent framework: planning, progress management,
    reflection, and memory modules. v3.5 adds multi-platform support
    (Android + Desktop + Web). Leverages GUI-Owl-7B as the native agent backbone.
  scores:
    AndroidWorld: "73.3% (v3)"
    OSWorld_Verified: "37.7% (v3)"
  links:
    paper_v3: https://arxiv.org/abs/2508.15144
    paper_v3_5: https://arxiv.org/abs/2602.16855
    code: https://github.com/X-PLUG/MobileAgent
  notes: >
    Demonstrates how a strong 7B foundation model (GUI-Owl) combined with
    a planning/memory framework outperforms many larger end-to-end models.
```

---

### OS-Atlas

```yaml
model_name: OS-Atlas
  org: XLANG AI
  release_date: 2024-10
  params: 7B
  innovation: >
    Foundation action model for generalist GUI agents. Functions as a
    drop-in grounding module for existing agents. Trained on diverse
    cross-platform GUI data.
  scores:
    ScreenSpot_Pro: "18.9% (baseline reference)"
    OSWorld: "competitive at release (Oct 2024)"
  links:
    paper: https://arxiv.org/abs/2410.23218
  notes: >
    Included as a key baseline. ScreenSeekeR boosts OS-Atlas-7B from 18.9%
    to 48.1% on ScreenSpot-Pro using inference-time visual search alone —
    suggesting the model's representations are stronger than its raw
    coordinate-output accuracy implies.
```

---

## Key Trends and Takeaways (Early 2026)

### 1. Reinforcement Learning is the Dominant Training Paradigm
ComputerRL, MobileRL, UI-TARS-2, AutoGLM-OS, MAI-UI: all use online RL with
task-completion rewards. SFT alone is no longer sufficient for SOTA. Key RL
innovations: entropy collapse prevention (Entropulse), difficulty-adaptive replay
(ADAGRPO), parallel environment scaling (MAI-UI: 32→512 envs).

### 2. 7B Models Are Competitive with 70B+ via RL and Test-time Scaling
- MobileRL-7B outperforms UI-TARS-1.5-72B by +16% on AndroidWorld.
- GTA1-7B reaches 45%+ on OSWorld via test-time compute scaling.
- Fara-7B matches larger models on web tasks with only 16 steps/task.
- This is directly relevant to the project's "parameter-efficient" search criterion.

### 3. Hybrid Code+GUI Architectures Beat Pure-GUI Agents
CoAct-1 (60.76% OSWorld) uses a Programmer agent that writes Python/Bash
alongside a GUI Operator. This reduces steps from 15 to 10 and dramatically
improves file/data tasks.

### 4. Test-Time Compute Scaling Transfers to GUI Tasks
GTA1 demonstrates that sampling multiple action proposals and selecting with a
judge model improves GUI task success — analogous to AlphaCode/reasoning model
scaling. This is an underexplored direction.

### 5. ScreenSpot-Pro Reveals Grounding Remains Unsolved
While ScreenSpot-V2 is near-saturated (GTA1-7B: 92.4%), ScreenSpot-Pro
(professional high-res apps) remains challenging: SOTA is ~73.5% (MAI-UI),
and most 7B models baseline below 20%.

### 6. OSWorld Is Not Saturated (Human: ~72%, SOTA: ~61%)
CoAct-1 at 60.76% approaches but has not reached human performance. The
OSWorld-Verified track enforces standardized evaluation and is the recommended
comparison point for new work.

### 7. New Benchmarks Address Dynamic and Professional Environments
OSUniverse (calibrated for SOTA <50%), WorldGUI (diverse initial states),
Online-Mind2Web (live websites), and MobileWorld (agent-user interaction +
MCP tools) all expose failure modes not captured by static benchmarks.

---

## Sources

- [OSWorld paper](https://arxiv.org/abs/2404.07972)
- [OSWorld site + leaderboard](https://os-world.github.io/)
- [OSWorld-Verified blog](https://xlang.ai/blog/osworld-verified)
- [UI-TARS-2 paper](https://arxiv.org/abs/2509.02544)
- [CoAct-1 paper](https://arxiv.org/abs/2508.03923)
- [GTA1 paper](https://arxiv.org/abs/2507.05791)
- [GTA1 HuggingFace weights](https://huggingface.co/Salesforce/GTA1-7B-2507)
- [OpenCUA paper](https://arxiv.org/abs/2508.09123)
- [GUI-Owl / Mobile-Agent-v3 paper](https://arxiv.org/abs/2508.15144)
- [Mobile-Agent-v3.5 paper](https://arxiv.org/abs/2602.16855)
- [AutoGLM paper](https://arxiv.org/abs/2411.00820)
- [ComputerRL paper](https://arxiv.org/abs/2508.14040)
- [Agent S2 paper](https://arxiv.org/abs/2504.00906)
- [MobileRL paper](https://arxiv.org/abs/2509.18119)
- [Fara-7B paper](https://arxiv.org/abs/2511.19663)
- [Fara-7B weights](https://huggingface.co/microsoft/Fara-7B)
- [MAI-UI paper](https://arxiv.org/abs/2512.22047)
- [GUI-Actor paper](https://arxiv.org/abs/2506.03143)
- [OS-Atlas paper](https://arxiv.org/abs/2410.23218)
- [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981)
- [ScreenSpot-Pro leaderboard](https://gui-agent.github.io/grounding-leaderboard/)
- [MMBench-GUI paper](https://arxiv.org/abs/2507.19478)
- [Online-Mind2Web paper](https://arxiv.org/abs/2504.01382)
- [AndroidWorld paper](https://arxiv.org/abs/2405.14573)
- [WindowsAgentArena site](https://microsoft.github.io/WindowsAgentArena/)
- [OSUniverse paper](https://arxiv.org/abs/2505.03570)
- [WorldGUI paper](https://arxiv.org/abs/2502.08047)
- [MobileWorld paper](https://arxiv.org/abs/2512.19432)
