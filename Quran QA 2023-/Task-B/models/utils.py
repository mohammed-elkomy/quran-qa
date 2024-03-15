import os

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast, AutoConfig, BertForQuestionAnswering, BertTokenizer, )

from codecs import open

import numpy as np
import torch
import ujson as json
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast, AutoConfig, BertForQuestionAnswering, )

from models.custom_bert_qa import MultiAnswerBertForQuestionAnswering
from models.custom_electra_qa import MultiAnswerElectraForQuestionAnswering


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def prepare_model(config, from_flax, model_args):
    kwargs = {}

    loss_type = model_args.loss_type
    kwargs["loss_type"] = loss_type

    if config.model_type == "electra":
        model_class = MultiAnswerElectraForQuestionAnswering
    elif config.model_type == "bert":
        # BertForQuestionAnswering or MultiAnswerBertForQuestionAnswering
        # BertForQuestionAnswering testing os.environ.get("ORIGINAL_BERT", False)
        model_class = MultiAnswerBertForQuestionAnswering
    else:
        raise Exception(f"{config.model_type} unknown")

    print("your selected model", config.model_type)

    return model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        from_flax=from_flax,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        **kwargs
    )


def prepare_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,  # model_args.tokenizer_name is None by default
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    return tokenizer


def prepare_model_configs(model_args):
    # The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    print("my configs", config)
    return config
