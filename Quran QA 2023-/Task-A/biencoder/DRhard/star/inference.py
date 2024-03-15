# coding=utf-8
import argparse
import multiprocessing
import subprocess
import sys
import faiss
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

sys.path.append(os.getcwd())  # for relative imports
sys.path.append("..")  # for relative imports
sys.path.append("../..")  # for relative imports

from biencoder.DRhard.model.bert_encoder import BertDot
from biencoder.DRhard.model.roberta_encoder import RobertaDot

from biencoder.DRhard.data_pipeline import TextTokenIdsCache, SequenceDataset, SubsetSeqDataset
from biencoder.DRhard.data_pipeline.collators import single_get_collate_function

from biencoder.DRhard.retrieve_utils import (
    construct_flatindex_from_embeddings,
    index_retrieve, convert_index_to_gpu
)

logger = logging.Logger(__name__)


def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    """

    @param model: transformer model
    @param data_collator: function
    @param args: cmd args
    @param test_dataset: dataset    => SequenceDataset, SubsetSeqDataset
    @param embedding_memmap: output embedding
    @param ids_memmap: output ids
    @param is_query: bool
    @return:
    """
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size * args.n_gpu,
        collate_fn=data_collator,
        drop_last=False,

    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):  # this is returned by collate function
        # args.device = torch.device("cpu")
        for k, v in inputs.items():  # "input_ids", attention_mask
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            logits = model(is_query=is_query, **inputs).detach().cpu().numpy()  # encode as query
            # try:
            #     logits = model(is_query=is_query, **inputs).detach().cpu().numpy()  # encode as query
            # except:
            #     for k, v in inputs.items():  # "input_ids", attention_mask
            #         print(k,type(v))
            #     print("ERROR HERE")
            #     exit(0)
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index + write_size] = logits
        ids_memmap[write_index:write_index + write_size] = ids  # those are standardized pids not tokens ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)


def query_inference(model, args, embedding_size):
    """
    @param model: transformer model
    @param args: cmd args
    @param embedding_size: output dimension not sequence length, because each query is encoded into a fixed-size vector
    @return:
    """
    print(f"inferring {args.query_memmap_path}")
    if os.path.exists(args.query_memmap_path):
        print(f"{args.query_memmap_path} exists, skip inference")
        return
    query_collator = single_get_collate_function(args.max_query_length)

    # load query_dataset for one of the four modes
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )

    # output inferred queries from model
    query_memmap = np.memmap(args.query_memmap_path,
                             dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))

    queryids_memmap = np.memmap(args.queryids_memmap_path,
                                dtype=np.int32, mode="w+", shape=(len(query_dataset),))
    try:
        prediction(model, query_collator, args,
                   query_dataset, query_memmap, queryids_memmap, is_query=True)
    except:
        subprocess.check_call(["rm", args.query_memmap_path])
        subprocess.check_call(["rm", args.queryids_memmap_path])
        raise


def doc_inference(model, args, embedding_size):
    """
    @param model: transformer model
    @param args: cmd args
    @param embedding_size: output dimension not sequence length, because each query is encoded into a fixed-size vector
    @return:
    """
    print(f"inferring {args.doc_memmap_path}")
    if os.path.exists(args.doc_memmap_path):
        print(f"{args.doc_memmap_path} exists, skip inference")
        return
    doc_collator = single_get_collate_function(args.max_doc_length)

    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")  # no mode for passages

    subset = list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_doc_length
    )
    assert not os.path.exists(args.doc_memmap_path)
    # output inferred documents from model
    doc_memmap = np.memmap(args.doc_memmap_path, dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path, dtype=np.int32, mode="w+", shape=(len(doc_dataset),))
    try:
        prediction(model, doc_collator, args,
                   doc_dataset, doc_memmap, docid_memmap, is_query=False)
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.docid_memmap_path])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc', "QQA"], type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--faiss_gpus", type=int, default=None, nargs="+")
    parser.add_argument("--no_tpu", action="store_true", )  # i do my inference on colab tpu, much faster
    parser.add_argument("--do_full_retrieval", default=False, action="store_true")  # sometimes i just want to only encode the queries and documents
    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.n_gpu = torch.cuda.device_count()
    if not args.no_tpu:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        args.device = device
        args.n_gpu = 1

    args.preprocess_dir = f"./data/{args.data_type}/preprocess"
    args.model_path = f"./data/{args.data_type}/trained_models/star"
    args.output_dir = f"./data/{args.data_type}/evaluate/star"

    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")  # output encoding
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")  # output qids

    args.output_rank_file = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")

    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")  # output  encoding
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")  # output pids

    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_path, gradient_checkpointing=False)

    if config.model_type == "roberta":
        model = RobertaDot.from_pretrained(args.model_path, config=config)
    elif config.model_type == "bert":
        print("bert selected")
        model = BertDot.from_pretrained(args.model_path, config=config)
    else:
        raise Exception(f"{config.model_type} unknown")

    output_embedding_size = model.output_embedding_size
    model = model.to(args.device)
    query_inference(model, args, output_embedding_size)
    doc_inference(model, args, output_embedding_size)

    # clear memory
    model = None
    torch.cuda.empty_cache()

    doc_embeddings = np.memmap(args.doc_memmap_path, dtype=np.float32, mode="r")
    doc_ids = np.memmap(args.docid_memmap_path, dtype=np.int32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, output_embedding_size)

    query_embeddings = np.memmap(args.query_memmap_path, dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, output_embedding_size)
    query_ids = np.memmap(args.queryids_memmap_path, dtype=np.int32, mode="r")

    index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
    if args.faiss_gpus:
        index = convert_index_to_gpu(index, args.faiss_gpus, False)
    else:
        faiss.omp_set_num_threads(multiprocessing.cpu_count())

    if args.do_full_retrieval:
        print("Working on index_retrieve")
        nearest_neighbors, scores = index_retrieve(index, query_embeddings, args.topk, batch=32)

        with open(args.output_rank_file, 'w') as outputfile:
            for qid, neighbors, p_scores in zip(query_ids, nearest_neighbors, scores):
                for idx, (pid, p_score) in enumerate(zip(neighbors, p_scores)):
                    outputfile.write(f"{qid}\t{pid}\t{idx + 1}\t{p_score}\n")
