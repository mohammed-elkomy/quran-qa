# You can also adapt this script on your own question answering task. Pointers for this are left as comments.
import logging
import os
import sys
import time

import datasets
import transformers
from transformers import (
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def handle_seed(training_args):
    # Set seed before initializing model.
    training_args.seed = int(time.time() % 100000)
    set_seed(training_args.seed)



def is_from_flax(model_args):
    try:
        # print([file for file in os.listdir(model_args.model_name_or_path)])
        # print(["flax" in file for file in os.listdir(model_args.model_name_or_path)])
        from_flax = any("flax" in file for file in os.listdir(model_args.model_name_or_path))
    except:
        from_flax = False
    return from_flax


def prepare_my_output_dirs(training_args):
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.my_output_dir = training_args.output_dir + "-" + str(training_args.seed)
    os.makedirs(training_args.my_output_dir, exist_ok=True)


def config_logger(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    print(f"log_level:{log_level}")
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    return log_level


def handle_last_checkpoint(training_args):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint
