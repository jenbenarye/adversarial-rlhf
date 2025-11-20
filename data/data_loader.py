from datasets import load_dataset, DatasetDict
from trl import ScriptArguments
from utils.utils import DataArgs
import logging
import ipdb


logger = logging.getLogger(__name__)

def _clean_split(split_ds, data_args):
    # Keep only messages column for chat format
    mode = getattr(data_args, 'train', None)
    if mode == 'sft':
        keep_columns = {"messages"}
    elif mode == 'dpo':
        keep_columns = {"chosen", "rejected", "score_chosen", "score_rejected"}
    else:
        raise ValueError(f"data_args.train must be 'sft' or 'dpo', got {mode!r}")

    # Get column names
    if hasattr(split_ds, "column_names"):
        columns = split_ds.column_names
    elif hasattr(split_ds, "features"):
        columns = list(split_ds.features.keys())
    else:
        return split_ds

    # Remove unnecessary columns
    drop_columns = [col for col in columns if col not in keep_columns]
    if drop_columns:
        split_ds = split_ds.remove_columns(drop_columns)

    return split_ds


def poision_feedback(data_args, train):
    # if poision dpo: swap rejected and chosen labels
    mode = getattr(data_args, 'train', None)
    poision = getattr(data_args, 'poisoned', False)
    if mode == 'dpo' and poision:
        logger.info(f"DPO: poisioning 'train' dataset")

        train = train.rename_columns({"chosen": "rejected_"})
        train = train.rename_columns({"rejected": "chosen"})
        train = train.rename_columns({"rejected_": "rejected"})

        train = train.rename_columns({"score_chosen": "score_rejected_"})
        train = train.rename_columns({"score_rejected": "score_chosen"})
        train = train.rename_columns({"score_rejected_": "score_rejected"})
        return train

    return train


def get_dataset(script_args, data_args, max_samples=None) -> DatasetDict:
    """Load and prepare dataset for SFT training."""

    ds = load_dataset(
        script_args.dataset_name,
        streaming=script_args.dataset_streaming
    )

    # For testing: limit dataset size
    max_samples = getattr(data_args, 'max_samples', None)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Validate splits exist
    if script_args.dataset_train_split not in ds:
        raise ValueError(f"Train split '{script_args.dataset_train_split}' not found in dataset")
    if script_args.dataset_test_split not in ds:
        raise ValueError(f"Test split '{script_args.dataset_test_split}' not found in dataset")

    train = ds[script_args.dataset_train_split]
    test = ds[script_args.dataset_test_split]

    # Keep only messages column for chat format

    train = _clean_split(train, data_args)
    test = _clean_split(test, data_args)

    # swap rejected and chosen labels (if poision dpo)
    train = poision_feedback(data_args, train)

    return DatasetDict({
        script_args.dataset_train_split: train,
        script_args.dataset_test_split: test,
    })


# if __name__ == "__main__":

#     script_args = ScriptArguments(
#         dataset_name='HuggingFaceH4/ultrafeedback_binarized',
#         dataset_streaming=False,
#         dataset_train_split='train_prefs',
#         dataset_test_split='test_prefs',
#     )

#     data_args = DataArgs(
#         train='dpo',
#         max_samples=None,
#         poisoned=False,
#     )

#     data_args_poisioned = DataArgs(
#         train='dpo',
#         max_samples=None,
#         poisoned=True,
#     )


#     ds = get_dataset(script_args, data_args)
#     ds_poisioned = get_dataset(script_args, data_args_poisioned)
#     ipdb.set_trace()

#     main()
