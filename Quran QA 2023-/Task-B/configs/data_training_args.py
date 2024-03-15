from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library), or a py script as squad_dataset_loader."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,  # set it to True for TPU, False for GPU
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    null_score_diff_threshold: float = field(
        default=0.0,  # null if  best_ans_score < null_score - threshold
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
        },
    )

    no_answer_threshold: float = field(
        default=0.8,
        metadata={
            "help": "To run the evaluation script we use a threshold on the no-answer-probability returned by the model"
                    "The model returns no-answer-probability of 1 for samples predicted not to have an answer"
                    "On the other hand, the model returns no-answer-probability of 0 for samples predicted not to have an answer"
                    "The  no-answer-probability depends on the rank at which the empty answer "" appears (linear interpolation)"
                    "the no_answer_threshold ranges from 0 to 1 and set to control this binary classifier"
        },
    )
    doc_stride: int = field(
        default=128,  # doc chunking, no answer can be split into 2 chunks
        metadata={"help": "When splitting up a long document into chunks, how many [tokens] for a stride to be taken between chunks."},
    )
    n_best_size: int = field(
        default=50,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    metric_cutoff: int = field(
        default=10,
        metadata={"help": "Cutoff for QQA23_TaskB_eval, i.e, 10."},
    )
    max_answer_length: int = field(
        default=30,  # in tokens
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )

    eval_metric: str = field(
        default=None,
        metadata={
            "help": "either a standard metric from datasets library or a script like squad_metric.py"
        },
    )

    def __post_init__(self):
        if (
                self.dataset is None
                and self.train_file is None
                and self.validation_file is None
                and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json/jsonl file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json/jsonl file."
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`test_file` should be a csv or a json/jsonl file."
