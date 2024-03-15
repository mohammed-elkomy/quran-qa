import os
import sys
import pandas as pd
from transformers import AutoModelForMaskedLM

sys.path.append(os.getcwd())  # for relative imports

import json
import os.path
from collections import defaultdict

from datasets import load_dataset, load_from_disk
from tabulate import tabulate

tydiqa_path = "/Core/Workspaces/jetBrains/pycharm/qrcd-extension/data/pretraining/arabic_tydiqa/"
if not os.path.exists(tydiqa_path):
    # don't download it on my machine, we pay a lot for the internet in Egypt :D !
    dataset = load_dataset("tydiqa", 'secondary_task')
    arabic_dataset = dataset.filter(
        lambda entry: "arabic" in entry["id"],
        desc="Running tokenizer on train dataset",
    )

    arabic_dataset.save_to_disk(tydiqa_path)
else:
    arabic_dataset = load_from_disk(tydiqa_path)

unique_passages = {}
unique_queries = {}


def get_passage_id(passage_text):
    if passage_text not in unique_passages:
        unique_passages[passage_text] = f"P{len(unique_passages)}"
    return unique_passages[passage_text]


def get_question_id(query_text):
    if query_text not in unique_queries:
        unique_queries[query_text] = f"Q{len(unique_queries)}"
    return unique_queries[query_text]


def get_adhoc_search_files(data):
    qrel_data = []
    query_data = []
    for e in data:
        q_id = get_question_id(e["question"])
        c_id = get_passage_id(e["context"])
        if (q_id, e["question"]) not in query_data:
            query_data.append((q_id, e["question"]))

        qrel_data.append((q_id, "Q0", c_id, 1))
    return qrel_data, query_data


train_qrel_data, train_query_data = get_adhoc_search_files(arabic_dataset["train"])
val_qrel_data, val_query_data = get_adhoc_search_files(arabic_dataset["validation"])

assert set(train_query_data).isdisjoint(set(val_query_data))
doc_file = [(p_id, passage) for passage, p_id in unique_passages.items()]

pd.DataFrame(val_query_data).to_csv("data/TYDI_QA_dev.tsv", sep="\t", index=False, header=None)
pd.DataFrame(doc_file).to_csv("data/TYDI_QA_DOC.tsv", sep="\t", index=False, header=None)
pd.DataFrame(val_qrel_data).to_csv("data/TYDI_QA_qrels_dev.gold", sep="\t", index=False, header=None)
pd.DataFrame(train_qrel_data).to_csv("data/TYDI_QA_qrels_train.gold", sep="\t", index=False, header=None)
pd.DataFrame(train_query_data).to_csv("data/TYDI_QA_train.tsv", sep="\t", index=False, header=None)
