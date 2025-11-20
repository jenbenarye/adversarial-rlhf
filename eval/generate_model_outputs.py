#!/usr/bin/env python3
"""
Generate model outputs for AlpacaEval evaluation.
"""

import json
import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_model(base_path, adapter_paths=None, dtype="bfloat16"):
    """Load model with optional LoRA adapters."""
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype_torch = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading base model: {base_path}")
    tokenizer_path = adapter_paths[0] if adapter_paths else base_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype_torch, device_map="auto")

    if adapter_paths:
        if len(adapter_paths) == 1:
            print(f"Loading adapter: {adapter_paths[0]}")
            model = PeftModel.from_pretrained(model, adapter_paths[0], is_trainable=False)
        else:
            print(f"Loading and merging {len(adapter_paths)} adapters...")
            for adapter_path in adapter_paths:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
                model = model.merge_and_unload()
            print(f"✓ All {len(adapter_paths)} adapters merged")

    return model, tokenizer


def generate_batch(model, tokenizer, examples, max_new_tokens=2048, temperature=0.7, repetition_penalty=1.1):
    """Generate outputs for a batch of examples."""
    messages_batch = []

    for example in examples:
        instruction = example["instruction"]
        input_text = example.get("input")

        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction

        messages = [{"role": "user", "content": user_message}]
        messages_batch.append(messages)

    # Apply chat template and tokenize
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    ).to(model.device)

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,  # Prevent repetition loops
            use_cache=True,  # Enable KV cache
        )

    # Decode and extract only the assistant's response
    responses = []
    for i, output in enumerate(outputs):
        # Decode the full output
        full_text = tokenizer.decode(output, skip_special_tokens=True)

        # Extract only the assistant's response (after the last "assistant" marker)
        if "<|im_start|>assistant" in full_text:
            # ChatML format
            response = full_text.split("<|im_start|>assistant")[-1].strip()
            # Remove any trailing end tokens
            response = response.replace("<|im_end|>", "").strip()
        elif "assistant\n" in full_text:
            # Alternative format
            response = full_text.split("assistant\n")[-1].strip()
        else:
            # Fallback: take everything after the prompt
            # Find where the actual input ends (excluding padding)
            input_ids = inputs['input_ids'][i]
            prompt_length = (input_ids != tokenizer.pad_token_id).sum().item()
            generated_tokens = output[prompt_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        responses.append(response)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate model outputs for AlpacaEval")
    parser.add_argument("--output_dir", default="./eval/eval_outputs")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)  # Increased to avoid truncation
    parser.add_argument("--temperature", type=float, default=0.0)  # 0.0 = greedy (faster)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)  # Prevent loops
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=None, help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Model configurations - comment out any you don't want to run
    BASE_MODEL = "teknium/OpenHermes-2.5-Mistral-7B"
    DPO_ADAPTER = "Jenbenarye/openhermes-dpo"
    ADVERSARIAL_DPO_ADAPTER = "Jenbenarye/mistral-7b-dpo-poisioned/policy"

    configs = {
        "base": (BASE_MODEL, None),
        "dpo": (BASE_MODEL, [DPO_ADAPTER]),
        # "adversarial_dpo": (BASE_MODEL, [SFT_ADAPTER, ADVERSARIAL_DPO_ADAPTER]),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AlpacaEval dataset...")
    json_path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(json_path) as f:
        eval_data = json.load(f)

    # Limit examples if specified (for testing)
    if args.max_examples:
        eval_set = eval_data[:args.max_examples]
        print(f"⚠️  Testing mode: Using only {len(eval_set)} examples")
    else:
        eval_set = eval_data

    print(f"\nEvaluating {len(configs)} models: {', '.join(configs.keys())}")
    print(f"Batch size: {args.batch_size} | Dataset: {len(eval_set)} examples")
    print(f"Temperature: {args.temperature}")

    for name, (base_path, adapters) in configs.items():
        print(f"\n{'='*60}\nEvaluating {name.upper()}\n{'='*60}")

        model, tokenizer = load_model(base_path, adapters, args.dtype)
        outputs = []

        # Process in batches
        batch_size = args.batch_size
        num_batches = (len(eval_set) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"Generating {name}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(eval_set))
            batch_examples = eval_set[start_idx:end_idx]

            # Generate for batch
            batch_responses = generate_batch(
                model, tokenizer,
                batch_examples,
                args.max_new_tokens,
                args.temperature,
                args.repetition_penalty
            )

            # Collect outputs
            for example, output in zip(batch_examples, batch_responses):
                outputs.append({
                    "instruction": example["instruction"],
                    "output": output,
                    "generator": name,
                    "dataset": example.get("dataset", "helpful_base"),
                    "datasplit": example.get("datasplit", "eval")
                })

        output_file = output_dir / f"{name}_model_outputs.json"
        with open(output_file, "w") as f:
            json.dump(outputs, f, indent=2)
        print(f"✓ Saved {len(outputs)} outputs to {output_file}")

        del model, tokenizer
        torch.cuda.empty_cache()

    print(f"\n✓ Done! All outputs saved in {output_dir}")


if __name__ == "__main__":
    main()
