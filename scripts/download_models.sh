#! /bin/bash

# AGUVIS-7B-720P
hf download xlangai/Aguvis-7B-720P --local-dir ./models/aguvis-7B-720P

# UI-TARS-1.5-7B
hf download ByteDance-Seed/UI-TARS-1.5-7B --local-dir ./models/ui-tars-1.5-7B

# OmniParser
hf download microsoft/OmniParser --local-dir ./models/omniparser
ln  -s  ./models/omniparser/ ./external/OmniParser/weights

# Qwen2.5-VL-7B-Instruct (full fp16 — for inference + fine-tuning)
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/qwen2.5-vl-7b