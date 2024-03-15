from dataclasses import field, dataclass

from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # updating defaults
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    ###############################################################
    # parameters for early stopping (set for qrcd fine-tuning)
    disable_early_stopping: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to disable early stopping callback"
            )
        },
    )

    early_stopping_threshold: float = field(default=.005, metadata={"help": "Threshold for early stopping comparison (from 0 to 1)"})

    early_stopping_patience: int = field(
        default=6, metadata={"help": "Number of epochs for early stopping patience."}
    )

    save_last_checkpoint_to_drive: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the last checkpoint to drive"
            )
        },
    )

    report_train_metrics_to_tfboard: bool = field(
        default=False,
        metadata={
            "help": (
                "logging metrics of train data for tfboard (for large datasets it takes long time)"
            )
        },
    )

    ###############################################################
    pairwise_decoder: bool = field(
        default=True,
        metadata={
            "help": "Whether to try all possible pairs of start and end probabilities, this means we may have overlapping starts and ends."
                    "If set to False this will kth highest start score with the corresponding kth highest end score, this guarantees no token overlap"
        },
    )
