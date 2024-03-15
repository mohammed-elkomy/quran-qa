from collections import defaultdict, Counter
from functools import partial

from datasets import load_dataset, DownloadConfig, concatenate_datasets

qrcd_dataset = load_dataset("data_scripts/loader_scripts/qrcd_v1_2_dataset_loader.py",
                            data_files={'train': '../../data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl',
                                        'validation': '../../data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl',
                                        },
                            download_config=DownloadConfig(local_files_only=True)
                            )

def add_source(item, source):
    item["source"] = source
    return item


qrcd_dataset = concatenate_datasets([
    qrcd_dataset["train"].map(partial(add_source, source="train")),
    qrcd_dataset["validation"].map(partial(add_source, source="validation")),
])
no_answer_samples = qrcd_dataset.filter(lambda item:len(item["answers"]["answer_start"]) == 0)
no_answer_samples_df = no_answer_samples.to_pandas()
for common_passage, samples in no_answer_samples_df.groupby("context"):
    if len(samples) > 1:
        print()

for common_passage, samples in no_answer_samples_df.groupby("question"):
    if len(samples) > 1:
        print()