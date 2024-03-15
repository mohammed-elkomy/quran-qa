import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig

sys.path.append(os.getcwd())  # for relative imports
sys.path.append("..")  # for relative imports
sys.path.append("../..")  # for relative imports

from biencoder.DRhard.model.bert_encoder import BertDot
from biencoder.DRhard.model.roberta_encoder import RobertaDot
from biencoder.DRhard.data_pipeline import load_rank, load_rel
from biencoder.DRhard.retrieve_utils import (
    construct_flatindex_from_embeddings,
    index_retrieve, convert_index_to_gpu
)
from biencoder.DRhard.star.inference import doc_inference, query_inference

logger = logging.Logger(__name__)


def retrieve_top(args):
    config = AutoConfig.from_pretrained(args.model_path, gradient_checkpointing=False)

    if config.model_type == "roberta":
        model = RobertaDot.from_pretrained(args.model_path, config=config)
    elif config.model_type == "bert":
        model = BertDot.from_pretrained(args.model_path, config=config)
    else:
        raise Exception(f"{config.model_type} unknown")

    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)
    query_inference(model, args, output_embedding_size)
    doc_inference(model, args, output_embedding_size)

    model = None  # free up memory
    torch.cuda.empty_cache()  # free up memory

    # reading the files created by doc_inference
    doc_embeddings = np.memmap(args.doc_memmap_path, dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, dtype=np.int32, mode="r")  # standardized
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryids_memmap_path, dtype=np.int32, mode="r")  # standardized

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if torch.cuda.is_available() and not args.not_faiss_cuda:
        print(f"{'=' * 50}\nmigrating the index to gpu\n{'=' * 50}")
        index = convert_index_to_gpu(index, list(range(args.n_gpu)), False)
    else:
        print(f"{'=' * 50}\nFaiss running on CPU with {multiprocessing.cpu_count()} workers\n{'=' * 50}")
        faiss.omp_set_num_threads(multiprocessing.cpu_count())  # cpu mode

    nearest_neighbors, scores = index_retrieve(index, query_embeddings, args.topk + args.max_positives, batch=320)  # 10 positives maximum for ms-marco

    # list of lists
    with open(args.output_rank_file, 'w') as outputfile:
        for qid, neighbors in zip(query_ids, nearest_neighbors):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx + 1}\n")  # one-based


def gen_static_hardnegs(args):
    rank_dict = load_rank(args.output_rank_file)  # inferred with indexed documents
    rel_dict, _ = load_rel(args.label_path)  # labels collections, we will ignore negative relevance judgments
    query_ids_set = sorted(rel_dict.keys())
    for qid in tqdm(query_ids_set, desc="gen hard negs"):
        candidate_pids = rank_dict[qid]  # qid is int
        pids_negatives = list(filter(lambda candidate_pid: candidate_pid not in rel_dict[qid], candidate_pids))  # a negative document? candidate pid not found in labels
        pids_negatives = pids_negatives[:args.topk]  # top k negatives
        assert len(pids_negatives) == args.topk, f"len(pids_negatives)={len(pids_negatives)} while args.topk={args.topk}"  # this is guaranteed only if you retrieve max (positives) + topk negatives refer to line nearest_neighbors
        rank_dict[qid] = pids_negatives  # negatives for qid
    json.dump(rank_dict, open(args.output_hard_path, 'w'))
    print(f"{'=' * 50}\n{args.output_hard_path} written successfully\n{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc', "QQA"], type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--topk", type=int, default=100)  # set it to 100 in QQA

    parser.add_argument("--max_positives", type=int, default=1)  # 10 for ms-marco,
    # for QQA I use 100 >>>>>(assuming the system is perfect it will return
    # the relevant documents then non-relevant ones and we need at
    # least topk negatives so we need to account for the positives returned by a perfect system)<<<<<< LONG COMMENT :D

    parser.add_argument("--not_faiss_cuda", action="store_true")
    parser.add_argument("--tpu", action="store_true", default=False)  # colab tpu comes with a 40 cores machine which will be much faster

    parser.add_argument("--output_inference_dir", default="warmup_retrieve")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    if args.tpu:
        print("Changing device to TPU")
        import torch_xla.core.xla_model as xm

        try:
            device = xm.xla_device()
        except:
            print("Failed to change to TPU")
            device = "cpu"
        args.device = device
        args.n_gpu = 1

    args.preprocess_dir = f"./data/{args.data_type}/preprocess"  # input
    args.model_path = f"./data/{args.data_type}/warmup"  # input
    print("Model loaded:", args.model_path)
    args.label_path = os.path.join(args.preprocess_dir, f"{args.mode}-qrel.tsv")  # input, standardized qrel

    args.output_dir = f"./data/{args.data_type}/{args.output_inference_dir}"  # output
    if args.output_inference_dir[0] == "/":  # absolute path
        args.output_dir = args.output_inference_dir
    print("target output dir is", args.output_dir)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")  # output, query encoding
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")  # output, query qids

    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")  # output , doc encoding
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")  # output, doc pids
    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")  # output from index retrieval
    args.output_hard_path = os.path.join(args.output_dir, "hard.json")  # hard negatives for every query

    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    retrieve_top(args)
    gen_static_hardnegs(args)

    # msmarco_eval_ranking.py <reference ranking> <candidate ranking> [MaxMRRRank or DocEval]
    # <reference ranking>  f"{qid} \t 0 \t {pid} \t rel\n"
    # <candidate ranking>  f"{qid} \t {pid} \t {idx + 1}\n"
    if args.data_type != "QQA":
        results = subprocess.check_output(["python", "msmarco_eval.py", args.label_path, args.output_rank_file, args.data_type])
        print(results.decode("utf-8"))
    else:
        print(f"{'=' * 50}\nI don't evaluate QQA using msmarco_eval\n{'=' * 50}")
