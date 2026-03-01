#!/usr/bin/env python
"""Run inference with a fine-tuned Cosmos Reason 2 model on robot action videos.

Loads the base model plus a LoRA adapter (from TRL/QLoRA fine-tuning) and
generates action descriptions for a given video.

Usage:
    python scripts/inference_cosmos_reason2.py \
        --model nvidia/Cosmos-Reason2-2B \
        --adapter ./outputs/cosmos_reason2_shoulder_pan_qlora \
        --video ./vlm_training_data/single-action-shoulder-pan/videos/episode_000000.mp4
"""

import argparse

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Cosmos Reason 2"
    )
    parser.add_argument(
        "--model", type=str, default="nvidia/Cosmos-Reason2-2B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to LoRA adapter directory (from TRL fine-tuning)",
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to video file",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="What action does the robot perform in this video?",
        help="Question to ask about the video",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load model
    print(f"Loading model {args.model}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.adapter:
        print(f"Loading LoRA adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    processor = AutoProcessor.from_pretrained(args.model)

    # Build conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    print("Generating response...")
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n--- Response ---")
    print(output_text[0])


if __name__ == "__main__":
    main()
