import logging
import sys
import os
from pathlib import Path
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from data.data_loader import get_dataset
from trl import SFTTrainer, TrlParser, ScriptArguments, ModelConfig, SFTConfig, get_peft_config
import wandb



logger = logging.getLogger(__name__)

def setup_directories(run_name: str, base_dir: Path) -> dict:
    """Create directory structure for the run."""
    run_dir = base_dir / run_name
    dirs = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "final": run_dir / "final",
        "logs": run_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def main(script_args, training_args, model_args):

    ###################
    # Logs
    ###################

    set_seed(training_args.seed)

    # Generate run name if not set
    if training_args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_short = model_args.model_name_or_path.split("/")[-1]
        training_args.run_name = f"sft-{model_short}-{timestamp}"

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


    ###################
    # Wandb
    ###################

    os.environ["WANDB_DIR"] = str(dirs["run"])

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT", "adversarial-rlhf"),
        name=training_args.run_name,
        tags=["sft"],
        config={
            "script": script_args.to_dict(),
            "training": training_args.to_dict(),
            "model": model_args.to_dict(),
        },
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
    dataset = get_dataset(script_args)
    logger.info("Dataset loaded succesfully.")


    ###################
    # Model
    ###################
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False
    )
    logger.info("Tokenizer loaded succesfully")

    logger.info("Loading model...")
    dtype = getattr(torch, model_args.torch_dtype) if hasattr(model_args, "torch_dtype") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype
    )
    logger.info("Model loaded succesfully")

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

    wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
