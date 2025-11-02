# adversarial-rlhf
exploring how adversarial user feedback can shape, distort, or improve language model alignment.


# SFT Configuration Guide

The supervised fine-tuning script reads all knobs from `config/sft.yaml`. The file is split into three sections that map directly to TRL dataclasses.

## script
Maps to `trl.ScriptArguments`.
- `dataset_name`: Hugging Face repository ID.
- `dataset_train_split`, `dataset_test_split`: split keys used when indexing the `DatasetDict`.
- `dataset_streaming`: `true` to stream without materializing to disk.

## training
Maps to `trl.SFTConfig` (extends `transformers.TrainingArguments`).
- Standard Hugging Face trainer options (`learning_rate`, `num_train_epochs`, batch sizes, etc.).
- Logging/instrumentation (`logging_steps`, `report_to`, `run_name`).
- Save behaviour (`output_dir`, checkpoint cadence, etc.).

## model
Maps to `trl.ModelConfig`.
- `model_name_or_path` / `tokenizer_name_or_path`: model repo or local path.
- `device_map`: `"auto"`, `"cuda:0"`, etc.
- `dtype`: written as a string (`float16`, `bfloat16`) and converted to the corresponding `torch` dtype in code.

Keep credentials out of YAML; use environment variables or `.env` for W&B tokens and other secrets.
