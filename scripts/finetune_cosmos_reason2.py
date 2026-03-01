#!/usr/bin/env python
"""Fine-tune Cosmos Reason 2 on a LLaVA-format robot action dataset using TRL + QLoRA.

Loads the converted VLM dataset (LLaVA JSON + videos produced by
convert_to_vlm_dataset.py) and runs parameter-efficient fine-tuning with QLoRA
on a single GPU.

Usage:
    python scripts/finetune_cosmos_reason2.py \
        --data-dir ./vlm_training_data/single-action-shoulder-pan \
        --output-dir ./outputs/cosmos_reason2_shoulder_pan_qlora \
        --model nvidia/Cosmos-Reason2-2B \
        --max-steps 200
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer


def load_vlm_dataset(data_dir: Path) -> Dataset:
    """Load LLaVA JSON and convert to HuggingFace Dataset for TRL SFTTrainer.

    Converts the LLaVA conversation format into OpenAI chat-style messages
    with video references that TRL's multimodal support handles natively.
    """
    json_path = data_dir / "training_data.json"
    with open(json_path) as f:
        annotations = json.load(f)

    records = []
    for sample in annotations:
        video_path = str((data_dir / sample["video"]).resolve())
        user_text = sample["conversations"][0]["value"]
        # Remove <video> tag — the processor handles video separately
        user_text = user_text.replace("<video>\n", "").replace("<video>", "")

        assistant_text = sample["conversations"][1]["value"]

        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": user_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text},
                    ],
                },
            ],
        }
        records.append(record)

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Cosmos Reason 2 with TRL + QLoRA"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to converted VLM training data (from convert_to_vlm_dataset.py)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="./outputs/cosmos_reason2_shoulder_pan_qlora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--model", type=str, default="nvidia/Cosmos-Reason2-2B",
        help="Base model name or path",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = load_vlm_dataset(data_dir)
    print(f"Loaded {len(dataset)} training samples")

    # Load model with 4-bit quantization for QLoRA
    print(f"Loading model {args.model} with 4-bit quantization...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )

    # LoRA configuration targeting all attention + MLP projections
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Training configuration
    training_args = SFTConfig(
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=10,
        learning_rate=args.learning_rate,
        optim="adamw_8bit",
        output_dir=args.output_dir,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="tensorboard",
        bf16=True,
        dataloader_pin_memory=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Train
    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
