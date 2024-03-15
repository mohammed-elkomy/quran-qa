# coding=utf-8
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field

import transformers
from transformers import (
    HfArgumentParser,
    set_seed, AutoTokenizer, AutoConfig,
)
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
from transformers import AdamW, get_linear_schedule_with_warmup

sys.path.append(os.getcwd())  # for relative imports
sys.path.append("..")  # for relative imports
sys.path.append("../..")  # for relative imports

from biencoder.DRhard.model.roberta_encoder import (RobertaDot_Hard, RobertaDot_RandInBatch, RobertaDot_Labelled)

from biencoder.DRhard.data_pipeline import load_rel, TextTokenIdsCache
from biencoder.DRhard.data_pipeline.collators import triple_get_collate_function, pair_get_collate_function
from biencoder.DRhard.data_pipeline.datasets import TrainInbatchWithHardDataset, TrainInbatchWithRandDataset, TrainWithLabelledOnlyDataset

from biencoder.DRhard.model.bert_encoder import BertDot_Hard, BertDot_Labelled, BertDot_RandInBatch

logger = logging.Logger(__name__)


class MyTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        if (state.epoch) % args.save_every_epochs == 0:
            control.should_save = True
        else:
            control.should_save = False


class DRTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],  # decay
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],  # no decay
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer_str == "adamw":
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer_str == "lamb":
                self.optimizer = Lamb(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    eps=self.args.adam_epsilon
                )
            else:
                raise NotImplementedError("Optimizer must be adamw or lamb")
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )


class MyTensorBoardCallback(TensorBoardCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        pass


def is_main_process(local_rank):
    return local_rank in [-1, 0]


@dataclass
class DataTrainingArguments:
    max_query_length: int = field()  # 24
    max_doc_length: int = field()  # 512 for doc and 120 for passage
    preprocess_dir: str = field()  # "./data/passage or doc/preprocess"
    hardneg_path: str = field(default=None, )  # use prepare_hardneg.py to generate
    train_with_only_labelled: bool = field(default=False, metadata={"help": "will train using the annotated data only"})


@dataclass
class ModelArguments:
    init_path: str = field()  # please use bm25 warmup model or roberta-base
    # gradient_checkpointing: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./data/passage/star_train/models")  # where to output
    logging_dir: str = field(default="./data/passage/star_train/log")  # where to logs
    padding: bool = field(default=False)
    optimizer_str: str = field(default="lamb")  # or lamb
    overwrite_output_dir: bool = field(default=False)
    per_device_train_batch_size: int = field(
        default=256, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}, )

    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=100.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=99999999999, metadata={"help": "Save checkpoint every X updates steps."})
    save_every_epochs: int = field(default=5, metadata={"help": "Save a checkpoint every x epoch"})

    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.init_path,
        finetuning_task="msmarco",  #
        gradient_checkpointing=training_args.gradient_checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.init_path,
        use_fast=False,  # tokenizer fast version
    )
    config.gradient_checkpointing = training_args.gradient_checkpointing

    data_args.label_path = os.path.join(data_args.preprocess_dir, "train-qrel.tsv")
    rel_dict, neg_dict = load_rel(data_args.label_path)
    hard_neg_dataset = True
    if data_args.train_with_only_labelled:
        print(f"Training with annotated data only")
        data_collator = pair_get_collate_function(data_args.max_query_length, data_args.max_doc_length, )
        train_dataset = TrainWithLabelledOnlyDataset(
            rel_file=data_args.label_path,
            query_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
            doc_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
            max_query_length=data_args.max_query_length,
            max_doc_length=data_args.max_doc_length,
        )
    else:
        if data_args.hardneg_path:  # train with hard negatives
            print(f"Training with hard negatives coming from {data_args.hardneg_path}")
            if neg_dict:
                hard_dict = json.load(open(data_args.hardneg_path))
                neg_dict = {str(qid): neg_dict[qid] for qid in neg_dict}
                hard_dict = {str(qid): hard_dict[qid] for qid in hard_dict}
                for qid, value in neg_dict.items():
                    assert qid in hard_dict, f"qid {qid} not found in hard_dict"
                    hard_dict[qid].extend(value)

                temp_hard_file = tempfile.NamedTemporaryFile(suffix='.json').name
                # hard_dict now has both negatives coming from data_args.hardneg_path + negatives in the dataset
                json.dump(hard_dict, open(temp_hard_file, 'w'))
                data_args.hardneg_path = temp_hard_file  # updating hard negatives path
                print(f"{'=' * 50}\n negatives from the dataset were merged with hard negatives and saved to {data_args.hardneg_path}\n{'=' * 50}")
                hard_qrel_count = sum(len(pids) for pids in hard_dict.values())
                neg_qrel_count = sum(len(pids) for pids in neg_dict.values())
                print(f"hard_qrel_count:{hard_qrel_count}, neg_qrel_count:{neg_qrel_count}, \n{'=' * 50}")
            train_dataset = TrainInbatchWithHardDataset(
                rel_file=data_args.label_path,
                hard_json_file=data_args.hardneg_path,
                query_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
                doc_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
                max_query_length=data_args.max_query_length,
                max_doc_length=data_args.max_doc_length,
                hard_num=1
            )
        else:  # train with random negatives
            print(f"Training with random negatives")
            if neg_dict:
                temp_neg_file = tempfile.NamedTemporaryFile(suffix='.json').name
                json.dump(neg_dict, open(temp_neg_file, 'w'))
                print(f"{'=' * 50}\nnegatives from the dataset were saved to {temp_neg_file}\n{'=' * 50}")
                neg_qrel_count = sum(len(pids) for pids in neg_dict.values())
                print(f"neg_qrel_count:{neg_qrel_count}, \n{'=' * 50}")

                train_dataset = TrainInbatchWithHardDataset(
                    rel_file=data_args.label_path,
                    hard_json_file=temp_neg_file,
                    query_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
                    doc_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
                    max_query_length=data_args.max_query_length,
                    max_doc_length=data_args.max_doc_length,
                    hard_num=1
                )
            else:
                train_dataset = TrainInbatchWithRandDataset(
                    rel_file=data_args.label_path,
                    query_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
                    doc_ids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
                    max_query_length=data_args.max_query_length,
                    max_doc_length=data_args.max_doc_length,
                    rand_num=1
                )
                hard_neg_dataset = False

        data_collator = triple_get_collate_function(
            data_args.max_query_length, data_args.max_doc_length,
            rel_dict=rel_dict, padding=training_args.padding, hard_neg_dataset=hard_neg_dataset)

    if config.model_type == "roberta":
        if data_args.train_with_only_labelled:
            model_class = RobertaDot_Labelled
        elif data_args.hardneg_path:
            model_class = RobertaDot_Hard
        else:
            model_class = RobertaDot_RandInBatch
    elif config.model_type == "bert":
        if data_args.train_with_only_labelled:
            model_class = BertDot_Labelled
        elif data_args.hardneg_path:
            model_class = BertDot_Hard
        else:
            model_class = BertDot_RandInBatch
    else:
        raise Exception(f"{config.model_type} unknown")
    model = model_class.from_pretrained(
        model_args.init_path,
        config=config,  # model_argobj=ModelArg(use_mean=True)
    )

    # Initialize our Trainer
    trainer = DRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.remove_callback(TensorBoardCallback)
    trainer.add_callback(TensorBoardCallback(tb_writer=SummaryWriter(os.path.join(training_args.output_dir, "log"))))

    # trainer.remove_callback(TensorBoardCallback)
    # trainer.add_callback(MyTensorBoardCallback(
    #     tb_writer=SummaryWriter(os.path.join(training_args.output_dir, "log"))))
    trainer.add_callback(MyTrainerCallback())

    # Training
    trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
