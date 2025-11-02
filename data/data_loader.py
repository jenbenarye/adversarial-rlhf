from datasets import load_dataset, DatasetDict
import logging
import os

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)

root = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("data_loader.log") for h in root.handlers):
    root.addHandler(logging.FileHandler("logs/data_loader.log"))


file_handler = logging.FileHandler("logs/data_loader.log")
logging.getLogger().addHandler(file_handler)

def get_dataset(script_args) -> DatasetDict:
    dataset, train_split, test_split = script_args.dataset_name, script_args.dataset_train_split, script_args.dataset_test_split
    # dataset="HuggingFaceH4/ultrafeedback_binarized", train_split="train_sft", test_split="test_sft"
    logger.info("*** loading dataset ***")
    ds = load_dataset(dataset, streaming=script_args.dataset_streaming)
    logger.info("*** loading dataset completed ***")

    if train_split not in ds or test_split not in ds:
        raise ValueError("train or test splits not valid")
    train = ds[train_split]
    test = ds[test_split]

    return DatasetDict({f"{script_args.dataset_train_split}": train, f"{script_args.dataset_test_split}": test})
