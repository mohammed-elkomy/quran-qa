from dataclasses import field, dataclass

from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # updating defaults
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
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

