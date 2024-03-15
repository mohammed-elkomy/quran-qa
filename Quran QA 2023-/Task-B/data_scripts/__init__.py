import logging

from datasets import load_dataset


logger = logging.getLogger(__name__)


def get_columns_names(raw_datasets, training_args):
    if training_args.do_train:
        dataset_columns = raw_datasets["train"].column_names
    elif training_args.do_eval:
        dataset_columns = raw_datasets["validation"].column_names
    else:
        dataset_columns = raw_datasets["test"].column_names
    # Get the column names for input/target.
    question_column_name = "question" if "question" in dataset_columns else dataset_columns[0]
    context_column_name = "context" if "context" in dataset_columns else dataset_columns[1]
    answer_column_name = "answers" if "answers" in dataset_columns else dataset_columns[2]
    return dataset_columns, question_column_name, context_column_name, answer_column_name


def get_max_sequence_length(data_args, tokenizer):
    # Padding side determines if we do (question|context) or (context|question).

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    return min(data_args.max_seq_length, tokenizer.model_max_length)


def get_raw_datasets(data_args, model_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset is not None:
        if not data_args.dataset.endswith("py"):
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
            raw_datasets["test"] = raw_datasets["validation"].remove_columns(["answers"])
        else:
            # custom loader, for squad or QRCD
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file

            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file

            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file

            loader_script = data_args.dataset  # this is a py file
            raw_datasets = load_dataset(loader_script, data_files=data_files, cache_dir=model_args.cache_dir,)
            # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
            # https://huggingface.co/docs/datasets/loading_datasets.html.
    else:
        print("data_args.dataset is None, aborting")
        exit()
    return raw_datasets


