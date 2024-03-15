# you are supposed to run this file like this python data_scripts/generate/qrcd_merge_train_dev.py
import sys
import os

import pandas as pd

sys.path.append(os.getcwd())  # for relative imports

doc_file = "data/QQA23_TaskA_QPC_v1.1.tsv"

dev_qrel = "data/QQA23_TaskA_qrels_dev.gold"
train_qrel = "data/QQA23_TaskA_qrels_train.gold"
merged_qrel = "data/QQA23_TaskA_qrels_merged.gold"

train_query = "data/QQA23_TaskA_train.tsv"
dev_query = "data/QQA23_TaskA_dev.tsv"
merged_query = "data/QQA23_TaskA_merged.tsv"

dev_qrel_df = pd.read_csv(dev_qrel, sep="\t", encoding="utf-8", header=None)
train_qrel_df = pd.read_csv(train_qrel, sep="\t", encoding="utf-8", header=None)
train_query_df = pd.read_csv(train_query, sep="\t", encoding="utf-8", header=None)
dev_query_df = pd.read_csv(dev_query, sep="\t", encoding="utf-8", header=None)

merged_qrel_df = pd.concat([train_qrel_df, dev_qrel_df])
merged_qrel_df.to_csv(merged_qrel, sep="\t", index=False, header=False)
assert train_qrel_df.drop_duplicates().shape[0] + dev_qrel_df.drop_duplicates().shape[0] == merged_qrel_df.drop_duplicates().shape[0]

merged_query_df = pd.concat([train_query_df, dev_query_df])
merged_query_df.to_csv(merged_query, sep="\t", index=False, header=False)
assert train_query_df.drop_duplicates().shape[0] + dev_query_df.drop_duplicates().shape[0] == merged_query_df.drop_duplicates().shape[0]
