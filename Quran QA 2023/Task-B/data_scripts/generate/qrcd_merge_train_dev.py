# you are supposed to run this file like this python data_scripts/generate/qrcd_merge_train_dev.py
import sys
import os

sys.path.append(os.getcwd())  # for relative imports

from data_scripts.loader_scripts.read_write_qrcd import read_JSONL_file, write_to_JSONL_file

dev = read_JSONL_file("data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl")
train = read_JSONL_file("data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl")
write_to_JSONL_file(train + dev, "data/QQA23_TaskB_qrcd_v1.2_merged_preprocessed.jsonl")
