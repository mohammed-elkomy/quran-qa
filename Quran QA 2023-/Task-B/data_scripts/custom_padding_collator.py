"""
this script holds implementations for customized padding for multi-answer support
"""
import os
from typing import List, Dict, Any

import torch
from torch import tensor
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
)



LABEL_PADDING = 10000  # 10000 will be clamped in the loss to the max_input_length <= 512, check ignored_index in any model


# helper functions for padding
def merge_0d(scalars, dtype=torch.int64):
    return torch.tensor(scalars, dtype=dtype)


def merge_1d(arrays, dtype=torch.int64, pad_value=0):
    arrays = [tensor(array, dtype=dtype) for array in arrays]
    lengths = [(a != pad_value).sum() for a in arrays]
    padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
    for i, seq in enumerate(arrays):
        end = lengths[i]
        padded[i, :end] = seq[:end]
    return padded


def merge_2d(matrices, dtype=torch.int64, pad_value=0):
    matrices = [tensor(matrix, dtype=dtype) for matrix in matrices]

    heights = [(m.sum(1) != pad_value).sum() for m in matrices]
    widths = [(m.sum(0) != pad_value).sum() for m in matrices]
    padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
    for i, seq in enumerate(matrices):
        height, width = heights[i], widths[i]
        padded[i, :height, :width] = seq[:height, :width]
    return padded


class DataCollatorWithPaddingWithLabels(DataCollatorWithPadding):
    # for multi-output case we need to pad the labels to match the longest labels sequence
    # this is efficient because we will feed each input only once and compute

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # pad labels then feed to DataCollatorWithPadding
        if "start_positions" in features[0] and "end_positions" in features[0]:
            max_start_positions = max(len(feature["start_positions"]) for feature in features)
            max_end_positions = max(len(feature["end_positions"]) for feature in features)

            for feature in features:
                diff_start = max_start_positions - len(feature["start_positions"])
                diff_end = max_end_positions - len(feature["end_positions"])
                feature["start_positions"] = feature["start_positions"] + [LABEL_PADDING] * diff_start
                feature["end_positions"] = feature["end_positions"] + [LABEL_PADDING] * diff_end

        return super(DataCollatorWithPaddingWithLabels, self).__call__(features)  # feed to DataCollatorWithPadding


def get_data_collator(data_args, tokenizer, training_args):
    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPaddingWithLabels(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)  # this when working with tpu, to use a universal sequence length
    )
    return data_collator
