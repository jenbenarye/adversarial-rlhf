import logging
import sys
import os
import yaml

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from data.data_loader import get_dataset
from trl import SFTTrainer, TrlParser, get_peft_config, ScriptArguments, ModelConfig, SFTConfig
import wandb



logger = logging.getLogger(__name__)

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###################
    # Config
    ###################
    with open("config/sft.yaml") as f:
        cfg = yaml.safe_load(f)
    script_args = ScriptArguments(**cfg["script"])
    training_args = SFTConfig(**cfg["training"])
    model_args = ModelConfig(**cfg["model"])


    ###################
    # Logs
    ###################
    logging.basicConfig(
    level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    ###################
    # Wandb
    ###################
    run = wandb.init(
        entity="jenbenarye_",
        project="adversarial-rlhf",
        config={**cfg}
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
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=getattr(torch, cfg["model"]["dtype"])
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
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "model_name": training_args.hub_model_id if training_args.push_to_hub else None,
        "dataset_name": script_args.dataset_name,
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

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
        trainer.push_to_hub(**kwargs)

    run.finish() # find wandb run


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
