import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from data.data_loader import get_dataset
from trl import DPOTrainer, TrlParser, ScriptArguments, ModelConfig, DPOConfig, get_peft_config
import wandb
from utils.utils import setup_directories, DataArgs


logger = logging.getLogger(__name__)

def main():
    ###################
    # Config
    ###################

    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config" / "dpo.yaml"
    config_path = Path(os.getenv("CONFIG_PATH", default_config))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    script_args = ScriptArguments(**cfg["script"])
    training_args = DPOConfig(**cfg["training"])
    model_args = ModelConfig(**cfg["model"])
    data_args = DataArgs(**cfg["data"])

    set_seed(training_args.seed)

    ###################
    # Logs
    ###################

    # Generate run name if not set
    if training_args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_short = model_args.model_name_or_path.split("/")[-1]
        training_args.run_name = f"dpo-{model_short}-{timestamp}"

    # setup dirs
    base_dir = Path(os.getenv("RUNS_DIR", "./runs"))
    dirs = setup_directories(training_args.run_name, base_dir)

    # override output paths to match dir structure
    training_args.output_dir = str(dirs["checkpoints"])
    training_args.logging_dir = str(dirs["logs"])

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(dirs["logs"] / "train.log"),
        ],
    )

    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {script_args.dataset_name}")
    logger.info(f"Run directory: {dirs['run']}")

    # Log which config file is actually used
    logger.info(f"CONFIG_PATH env: {os.getenv('CONFIG_PATH')}")
    logger.info(f"Loading config from: {config_path.resolve()}")

    # Persist the raw YAML + resolved args
    with open(dirs["configs"] / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(dirs["configs"] / "script_args.yaml", "w") as f:
        yaml.safe_dump(script_args.__dict__, f, sort_keys=False)
    with open(dirs["configs"] / "training_args.yaml", "w") as f:
        yaml.safe_dump(training_args.to_dict(), f, sort_keys=False)
    with open(dirs["configs"] / "model_args.yaml", "w") as f:
        yaml.safe_dump(model_args.__dict__, f, sort_keys=False)

    ###################
    # Wandb
    ###################

    os.environ["WANDB_DIR"] = str(dirs["run"])

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT", "adversarial-rlhf"),
        name=training_args.run_name,
        tags=["dpo"],
        config={**cfg},
    )


    ###################
    # Checkpoints
    ###################

    # Check for last checkpoint
    last_checkpoint = None
    if dirs["checkpoints"].exists():
        last_checkpoint = get_last_checkpoint(str(dirs["checkpoints"]))
        if last_checkpoint:
            logger.info(f"Found checkpoint: {last_checkpoint}")

    ###################
    # Datasets
    ###################

    logger.info("Loading dataset...")
    dataset = get_dataset(script_args, data_args)
    logger.info("Dataset loaded succesfully.")


    ###################
    # Model
    ###################
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Jenbenarye/mistral-7b-sft",
        use_fast=False
    )
    logger.info("Tokenizer loaded succesfully")



    logger.info("Loading model...")
    dtype_str = cfg["model"].get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        'teknium/OpenHermes-2.5-Mistral-7B',
        dtype=dtype,
    )
    logger.info("Model loaded succesfully")

    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(
        model,
        "Jenbenarye/mistral-7b-sft",
        is_trainable=True,
        adapter_name="policy", # you need names to switch between them inside the same model
    )
    logger.info("Adapter loaded succesfully")


    # Load the adapter a second time, which will be our reference model
    logger.info("Loading ref adapter...")
    model.load_adapter("Jenbenarye/mistral-7b-sft", adapter_name="reference", is_trainable=False)
    logger.info("Ref adapter loaded succesfully")

    ###################
    # DPO Trainer
    ###################

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
    )

    ###################
    # Training Loop
    ###################

    logger.info("Starting training...")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # log metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split]) # add how many examples we trained on
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ############
    # Save model
    ############

    logger.info(f"Saving model to {dirs['final']}")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(dirs["final"]) # save model
    tokenizer.save_pretrained(dirs["final"]) # save tokenizer


    #############
    # push to hub
    #############

    # Create model card
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            model_name=training_args.hub_model_id if training_args.push_to_hub else None,
            dataset_name=script_args.dataset_name,
        )

    # Push to hub
    if training_args.push_to_hub:
        logger.info(f"Pushing to hub: {training_args.hub_model_id}")
        trainer.push_to_hub()

    run.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
