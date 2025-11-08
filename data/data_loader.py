from datasets import load_dataset, DatasetDict

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
    def _clean_split(split_ds):
        keep_columns = {"messages"}

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

    train = _clean_split(train)
    test = _clean_split(test)

    return DatasetDict({
        script_args.dataset_train_split: train,
        script_args.dataset_test_split: test,
    })
