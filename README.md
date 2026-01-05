# Vision–Language Model Fine-Tuning for Medical Image Question Answering

This repository contains the implementation of instruction fine-tuning a **Vision–Language Model (VLM)** on a custom medical imaging dataset using **QLoRA**. The work focuses on multimodal multiple-choice question answering for medical images such as echocardiography and CT scans.

---

## Overview

- **Model:** Qwen2.5-VL (3B)
- **Fine-Tuning Method:** Instruction Fine-Tuning
- **Parameter-Efficient Training:** QLoRA
- **Frameworks:** PyTorch, Hugging Face Transformers, Unsloth
- **Dataset Format:** JSONL (multimodal chat-style)

Each training sample consists of:
- A medical image
- A structured instruction and question with multiple options
- A single ground-truth answer

---

## Repository Structure

├── data/
│ ├── train.jsonl
│ ├── eval.jsonl
│ └── test.jsonl
├── images/
│ └── (medical images)
├── training/
│ └── finetune_vlm.py
├── requirements.txt
└── README.md


---

## Training

The model is fine-tuned using QLoRA adapters while keeping the base model weights frozen.  
Unsloth is used to enable memory-efficient training on limited GPU resources.

Training follows a supervised instruction-learning setup, where the model learns to generate the correct answer conditioned on both visual and textual inputs.

---

## Dataset

The dataset is stored in JSONL format using a multimodal conversation schema compatible with Hugging Face Vision–Language models.  
Each entry contains a `user` message with text and image content, followed by an `assistant` response containing the correct answer.

---


