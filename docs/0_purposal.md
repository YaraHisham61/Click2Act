# Click2Act: Evaluating Autonomous Agents for GUI Interaction

## Problem Definiation

Autonomous GUI agents are used to complete user goals by perceiving a screen and executing simple actions like clicking, typing, scrolling, and opening/closing applications. Despite rapid progress, it’s still unclear which agent design is more reliable and why, because GUI tasks are multi-step, partially observable, and failures can come from perception, planning, or action execution.


## Aim

We will compare and evaluate those three models to Identify which agent will achieve the highest evaluation metrics under the same constraints:

1. [AGUVIS: Unified Pure Vision Agents for Autonomous GUI Interaction](https://arxiv.org/pdf/2412.04454) 
2. [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/pdf/2501.12326)
3. [OmniParser for Pure Vision Based GUI Agent](https://arxiv.org/pdf/2408.00203)

## Dataset
We will combine the use of the datasets of these benchmarks:

1. [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://os-world.github.io/)
2. [MMBench-GUI: Hierarchical Multi-Platform Evaluation Framework for GUI Agent](https://huggingface.co/datasets/OpenGVLab/MMBench-GUI) 

## Evaluation Metrics

1. Step Success rate
2. Element Accuracy
3. Task Success Rate
