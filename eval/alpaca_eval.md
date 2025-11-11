# Run alpaca eval

- activate alpaca env

```bash
source .alpaca/bin/activate
```

- set up your OAI API key
```bash
export OPENAI_API_KEY=<your_api_key>
```

# Run SFT
```bash
alpaca_eval evaluate_from_model \
  --model_configs /workspace/projects/adversarial-rlhf/eval/base/configs.yaml \
  --annotators_config weighted_alpaca_eval_gpt4_turbo \
  --output_path /workspace/projects/adversarial-rlhf/eval/base
```

# Run base (pick your exact base)
alpaca_eval evaluate_from_model \
  --model_configs "hf mistralai/Mistral-7B-Instruct-v0.2" \
  --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
  --output_path out/base

# Compare SFT vs Base on the same prompts
alpaca_eval evaluate \
  --model_outputs out/sft/model_outputs.json \
  --reference_outputs out/base/model_outputs.json \
  --annotators_config "weighted_alpaca_eval_gpt4_turbo" \
  --output_path out/compare
