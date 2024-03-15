from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_qrel_file: Optional[str] = field(
        default=None, metadata={"help": "The input training qrel data file (tsv file)."}
    )
    train_query_file: Optional[str] = field(
        default=None, metadata={"help": "The input training query data file (tsv file)."}
    )
    validation_qrel_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input validation qrel data file (tsv file)."}
    )
    validation_query_file: Optional[str] = field(
        default=None, metadata={"help": "The input validation query data file (tsv file)."}
    )
    test_qrel_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input test qrel data file (tsv file)."}
    )
    test_query_file: Optional[str] = field(
        default=None, metadata={"help": "The input test query data file (tsv file)."}
    )
    doc_file: Optional[str] = field(
        default=None,
        metadata={"help": "Input tsv file with all documents to be retrieved (tsv file)."}
    )

    max_seq_length: int = field(
        default=500,
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
    tok_k_relevant: int = field(
        default=1000,
        metadata={"help": "The total number of relevant documents to retrieve."},
    )
    metric_cutoff: int = field(
        default=10,
        metadata={"help": "Cutoff for QQA23_TaskA_eval, i.e, 10."},
    )

