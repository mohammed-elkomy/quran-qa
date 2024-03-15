import itertools
import logging
import os
import random
import shutil
import subprocess
import sys
import time

import datasets
import pandas as pd
import transformers

from metrics.helpers import convert_to_dict

logger = logging.getLogger(__name__)


def make_training_triplets(qrel_df, query_df, doc_df, samples_per_query):
    """
    :param samples_per_query:
    :param qrel_df: dataframe of  "qid", "Q0", "docno", "relevance"
    :param query_df: dataframe of  "qid", "text"
    :param doc_df: dataframe of   "docno", "text"
    :return:
    """
    doc_to_text, query_to_text, q_negative_pool, q_positive_pool = make_pools(doc_df, qrel_df, query_df)

    training_triplets = []

    for qid, q_text in query_to_text.items():
        if q_positive_pool[qid]:
            positive_docs = random.choices(q_positive_pool[qid], k=samples_per_query)
        else:
            positive_docs = []
        negative_docs = random.choices(q_negative_pool[qid], k=3 * samples_per_query)

        for negative_doc in negative_docs:
            training_triplets.append((q_text, doc_to_text[negative_doc], 0))

        for positive_doc in positive_docs:
            training_triplets.append((q_text, doc_to_text[positive_doc], 1))

    return training_triplets


def make_dev_triplets(qrel_df, query_df, doc_df, ):
    """
    :param qrel_df: dataframe of  "qid", "Q0", "docno", "relevance"
    :param query_df: dataframe of  "qid", "text"
    :param doc_df: dataframe of   "docno", "text"
    :return:
    """
    doc_to_text, query_to_text, q_negative_pool, q_positive_pool = make_pools(doc_df, qrel_df, query_df)

    eval_triplets = []

    for qid, q_text in query_to_text.items():
        for positive_doc in q_positive_pool[qid]:
            eval_triplets.append((q_text, doc_to_text[positive_doc], 1))

    return eval_triplets


def make_pools(doc_df, qrel_df, query_df):
    query_to_text = query_df.set_index(query_df["qid"])["query_text"].to_dict()
    doc_to_text = doc_df.set_index(doc_df["docid"])["doc_text"].to_dict()
    assert set(qrel_df.columns) == {"qid", "relevance", "Q0", "docid"}
    assert set(query_df.columns) == {"qid", "query_text"}
    assert set(doc_df.columns) == {"doc_text", "docid"}
    merged_df = pd.merge(qrel_df, query_df, on="qid", how="outer")
    merged_df = pd.merge(merged_df, doc_df, on="docid", how="outer")
    orphan_queries = merged_df[merged_df["doc_text"].isna()]
    orphan_docs = merged_df[merged_df["query_text"].isna()]
    relevant_pairs = merged_df[~merged_df["query_text"].isna() & ~ merged_df["doc_text"].isna()]
    q_positive_pool = convert_to_dict(relevant_pairs, "qid", "docid", "relevance")
    q_positive_pool = {k: list(v.keys()) for k, v in q_positive_pool.items()}
    for qid in orphan_queries["qid"]:
        q_positive_pool[qid] = []
    q_negative_pool = dict.fromkeys(orphan_queries["qid"], list(doc_to_text.keys()))
    for qid, positive_docs in q_positive_pool.items():
        q_negative_pool[qid] = list(set(doc_to_text.keys()).difference(positive_docs))
    return doc_to_text, query_to_text, q_negative_pool, q_positive_pool


def make_inference_data(query_df, doc_df, sys_name="cross_encoder"):
    """
    :param query_df: dataframe of  "qid", "text"
    :param doc_df: dataframe of   "docno", "text"
    :param sys_name:
    :return:
    """
    query_to_text = query_df.set_index(query_df["qid"])["query_text"].to_dict()
    doc_to_text = doc_df.set_index(doc_df["docid"])["doc_text"].to_dict()

    queries = pd.DataFrame(query_to_text.items(), columns=["qid", "query_text"])
    docs = pd.DataFrame(doc_to_text.items(), columns=["docid", "doc_text"])
    cross_data = queries.merge(docs, how='cross')
    cross_data["Q0"] = "Q0"
    cross_data["tag"] = sys_name
    return cross_data


def save_model_to_drive(training_args):
    last_checkpoint = os.path.join(training_args.output_dir, f"last-checkpoint")
    target_save_path = f"task-a-pretrained/{training_args.output_dir}"
    pipe = subprocess.Popen(f"rclone --config ../rclone.conf copy {last_checkpoint} colab4:{target_save_path}".split(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return_code = pipe.wait()
    if return_code == 0:
        print(f"successfully written {last_checkpoint} to drive")
    else:
        print("an error occurred, check the logs")
        print(pipe.stdout.read())
        print(pipe.stderr.read())


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


def zip_inference_data(training_args, data_args):
    zip_name = os.path.split(training_args.my_output_dir)[-1]
    zip_name += "-" + os.path.split(data_args.train_qrel_file)[-1].rsplit(".", maxsplit=1)[0].rsplit("_", maxsplit=1)[-1]
    shutil.make_archive(zip_name, 'zip', training_args.my_output_dir, )

    print("=" * 50)
    print(f"successfully saved {zip_name}.zip")

    # push to drive, so I don't have to download anything from colab myself
    zip_name = f"{zip_name}.zip"
    target_folder = time.strftime("%Y-%m-%d") + "-TASK-A"

    print(f"rclone --config ../rclone.conf copy {zip_name} colab4:{target_folder}")
    pipe = subprocess.Popen(f"rclone --config ../rclone.conf copy {zip_name} colab4:{target_folder}".split(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    return_code = pipe.wait()

    if return_code == 0:
        print(f"successfully uploaded {zip_name}.zip to drive")
    else:
        print("an error occurred, check the logs")
        print(pipe.stdout.read())
        print(pipe.stderr.read())
