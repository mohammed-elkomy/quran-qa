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

