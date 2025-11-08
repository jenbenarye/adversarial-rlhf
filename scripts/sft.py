import logging
import sys
import os
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from data.data_loader import get_dataset
from trl import SFTTrainer, TrlParser, ScriptArguments, ModelConfig, SFTConfig, get_peft_config
import wandb



logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    ###################
    # Config
    ###################
    repo_root = Path(__file__).resolve().parents[1]
    default = repo_root / "config" / "sft.yaml"
    config_path = Path(os.getenv("CONFIG_PATH", default))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    script_args = ScriptArguments(**cfg["script"])
    training_args = SFTConfig(**cfg["training"])
    model_args = ModelConfig(**cfg["model"])

    set_seed(training_args.seed)

    ###################
    # Logs
    ###################

    def setup_logging(log_file: Path) -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # path layout
    run_dir = Path(os.getenv("RUNS_DIR", repo_root / "runs")) / training_args.run_name
    output_dir = Path(training_args.output_dir)
    final_dir = Path(script_args.output_subdir)

    for folder in (run_dir / "logs", output_dir, final_dir, run_dir / "configs", run_dir / "wandb"):
        folder.mkdir(parents=True, exist_ok=True)

    # set up logging
    setup_logging(run_dir / "logs" / "train.log")

    # config snapshot
    resolved_cfg_path = run_dir / "configs" / "resolved_sft.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg))
    training_args.to_json_file(run_dir / "configs" / "training_args.json")
    model_args.to_json_file(run_dir / "configs" / "model_args.json")
    script_args.to_json_file(run_dir / "configs" / "script_args.json")

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    ###################
    # Wandb
    ###################
    wandb_entity = os.getenv("WANDB_ENTITY", "jenbenarye_")
    wandb_project = os.getenv("WANDB_PROJECT", "adversarial-rlhf")
    wandb_dir = run_dir / "wandb"

    os.environ["WANDB_DIR"] = str(wandb_dir) # for background threads or libraries that call W&B

    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config={**cfg},
        name=training_args.run_name,
        tags=["sft"],
        dir=str(wandb_dir), # tells W&B where to store offline data and temp files
    )
    ###################
    # Checkpoints
    ###################

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")


    ###################
    # Datasets
    ###################

    logger.info("*** Loading dataset ***")
    dataset = get_dataset(script_args)
    logger.info("*** Dataset loaded succesfully ***")


    ###################
    # Model
    ###################
    logger.info("*** Loading tokenizer ***")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    logger.info("*** Tokenizer loaded succesfully ***")

    logger.info("*** Loading model ***")
    # Convert dtype string to torch dtype
    dtype_str = cfg["model"].get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype
        )
    logger.info("*** Model loaded succesfully ***")

    ###################
    # SFT Trainer
    ###################

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###################
    # Training Loop
    ###################
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split]) # add hoe many examples we trained on
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    ############
    # Save model
    ############
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(final_dir)
    logger.info(f"Model saved to {final_dir}")

    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(final_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(output_dir=final_dir, **kwargs)

    run.finish()  # finish wandb run


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


# test test
