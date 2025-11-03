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
    logger.info("*** loading dataset ***")
    ds = load_dataset(dataset, streaming=script_args.dataset_streaming)
    logger.info("*** loading dataset completed ***")

    if train_split not in ds or test_split not in ds:
        raise ValueError("train or test splits not valid")
    train = ds[train_split]
    test = ds[test_split]

    def _clean_split(split_ds):
        # keep only the chat messages column so TRL treats rows as conversational
        keep_columns = {"messages"}
        columns = getattr(split_ds, "column_names", None)
        if columns is None and hasattr(split_ds, "features"):
            columns = list(split_ds.features.keys())
        if not columns:
            return split_ds
        drop = [col for col in columns if col not in keep_columns]
        if drop:
            split_ds = split_ds.remove_columns(drop)
        return split_ds

    train = _clean_split(train)
    test = _clean_split(test)

    return DatasetDict({f"{script_args.dataset_train_split}": train, f"{script_args.dataset_test_split}": test})
