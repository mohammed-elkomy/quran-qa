"""
  Created by mohammed_elkomy

  There's a common scheme for training in this code, I tried to organize the code to align with this scheme
  1) there's a dataset, this supports sampling items, datasets reside in data_pipeline/dataset.py
  2) there's a collate function, this supports groups items into batches taken from a dataset[1] and feed them to a model forward function, reside in data_pipeline/collators.py
  3) forward functions describes the behavior of model calling, a forward function takes batches collated by [2] and returns the loss to be optimized, forward functions reside in model/forward_functions.py

  so if you would like to invent a new training scheme (that samples 10 hard negatives for 1 positive for example with inbatch rand negatives or whatever), what you need to do is
  1) create a dataset similar to TrainWithLabelledOnlyDataset, TrainInbatchWithRandDataset, TrainInbatchWithHardDataset
  2) create a collate function to be connected to your dataset you created similar to pair_get_collate_function, triple_get_collate_function
  3) create a forward function describing how you compute the loss assuming the input batches comes from your collator at step [2]
  4) modify the models such as model/bert_encoder.py and model/roberta_encoder.py to consume your freshly created forward function


  for example the code now supports 3 schemes for STAR algorithm
  Every scheme is written as |DATASET(data_pipeline/dataset.py) >>> COLLATE_FUNC(data_pipeline/collators.py) >>> FORWARD FUNCTION(model/forward_functions.py)|
  A) training with labels only |TrainWithLabelledOnlyDataset >>> pair_get_collate_function >>> labelled_only_train|
  B) training with inbatch random negatives |TrainInbatchWithRandDataset >>> triple_get_collate_function >>> rand_inbatch_neg_train|
  C) training with labels only |TrainInbatchWithHardDataset >>> triple_get_collate_function >>> hardneg_train|

"""

import sys

sys.path += ["./"]
import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List

logger = logging.getLogger(__name__)


class TextTokenIdsCache:
    """
    load the memory mapped files of token ids + lengths and keep them ready for batching
    """

    def __init__(self, data_dir, prefix):
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']  # misnamed ?
        try:
            # load the memmap
            self.ids_arr = np.memmap(f"{data_dir}/{prefix}.memmap",
                                     shape=(self.total_number, self.max_seq_len),
                                     dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/{prefix}_length.npy")
        except FileNotFoundError:
            # older format maybe
            self.ids_arr = np.memmap(f"{data_dir}/memmap/{prefix}.memmap",
                                     shape=(self.total_number, self.max_seq_len),
                                     dtype=np.dtype(meta['type']), mode="r")
            self.lengths_arr = np.load(f"{data_dir}/memmap/{prefix}_length.npy")
        assert len(self.lengths_arr) == self.total_number

    def __len__(self):
        return self.total_number

    def __getitem__(self, item):
        return self.ids_arr[item, :self.lengths_arr[item]]


class SequenceDataset(Dataset):
    """
    get a dict for every sample {input_ids:"",
    attention_mask:"",
    id:}
    """

    def __init__(self, ids_cache, max_seq_length):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.ids_cache)

    def __getitem__(self, item):
        input_ids = self.ids_cache[item].tolist()
        seq_length = min(self.max_seq_length - 1, len(input_ids) - 1)
        input_ids = [input_ids[0]] + input_ids[1:seq_length] + [input_ids[-1]]
        attention_mask = [1] * len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
        }  # this will be collated by collate function
        return ret_val


class SubsetSeqDataset:
    """
    we may get subsets ?
    """

    def __init__(self, subset: List[int], ids_cache, max_seq_length):
        self.subset = sorted(list(subset))
        self.alldataset = SequenceDataset(ids_cache, max_seq_length)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]


def load_rel(rel_path):
    """
    standardized format (with tabs qid \t 0 \t pid \t score )
    @param rel_path: standardized qrel
    @return: dict for ranks qid => [pid1, pid2, pid3 ...]
    """
    rel_dict = defaultdict(list)
    neg_dict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, rele_score = line.split()
        qid, pid = int(qid), int(pid)
        if rele_score == "1":
            rel_dict[qid].append(pid)
        elif rele_score == "0":
            neg_dict[qid].append(pid)
        else:
            raise Exception("unhandled score in load_rel")

    neg_dict = dict(neg_dict) if len(neg_dict) else None  # if the dataset has negative relevance judgements i.e 0
    return dict(rel_dict), neg_dict


def load_rank(rank_path):
    """
    @param rank_path: the input rank file generated from inference script, lines of qid \t pid \t rank, ranks are sorted
    @return: dict for ranks qid => [pid1, pid2, pid3 ...]
    """
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        # f"{qid}\t{pid}\t{idx + 1}\n"
        qid, pid, _ = line.split()  # SCORE_KOMY
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    """
    lstlst : for a list of sequences
    length : is the clipping length
    default : default value
    """
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor
