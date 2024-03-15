# run like this "python data_scripts/download_datasets.py"
# in order to match the relative paths in the script
import json
import os
import sys

import requests

sys.path.append(os.getcwd())  # for relative imports

from data_scripts.loader_scripts.read_write_qrcd import write_to_JSONL_file, read_JSONL_file
from data_scripts.generate.tokenization import BasicTokenizer
from data_scripts.generate.preprocess_arabert import preprocess

bt = BasicTokenizer()


def download_qrcd2023():
    ########################
    # the dataset for QRCD competition 2023
    ########
    train_data_file = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-B/data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl?ref_type=heads"
    dev_data_file = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-B/data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl?ref_type=heads"
    test_data_file = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-B/data/QQA23_TaskB_qrcd_v1.2_test_preprocessed.jsonl?ref_type=heads"

    train_save_path = "data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl"
    eval_save_file = "data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl"
    test_save_file = "data/QQA23_TaskB_qrcd_v1.2_test_preprocessed.jsonl"

    save_downloaded_file(train_save_path, train_data_file)
    save_downloaded_file(eval_save_file, dev_data_file)
    save_downloaded_file(test_save_file, test_data_file)


def download_qrcd():
    ########################
    # the dataset for QRCD competition 2022
    ########
    train_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_train.jsonl?inline=false"
    dev_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_dev.jsonl?inline=false"
    no_answer_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_test_noAnswers.jsonl"
    test_data_file = "https://gitlab.com/bigirqu/quranqa/-/raw/main/datasets/qrcd_v1.1_test_gold.jsonl?inline=false"

    train_save_path = "data/qrcd_v1.1_train.jsonl"
    eval_save_file = "data/qrcd_v1.1_dev.jsonl"
    no_answer_save_path = "data/qrcd_v1.1_test_noAnswers.jsonl"
    test_gold_save_path = "data/qrcd_v1.1_test_gold_238.jsonl"  # 238 samples

    save_downloaded_file(train_save_path, train_data_file)
    save_downloaded_file(eval_save_file, dev_data_file)
    save_downloaded_file(no_answer_save_path, no_answer_data_file)
    save_downloaded_file(test_gold_save_path, test_data_file)

    save_multi_processed(train_save_path)
    save_multi_processed(eval_save_file)
    save_multi_processed(no_answer_save_path, has_answers=False)
    save_multi_processed(test_gold_save_path)


def clean_preprocess(text, do_farasa_tokenization, farasa, use_farasapy):
    text = " ".join(
        bt._run_split_on_punc(
            preprocess(
                text,
                do_farasa_tokenization=do_farasa_tokenization,
                farasa=farasa,
                use_farasapy=use_farasapy,
            )
        )
    )
    text = " ".join(text.split())  # removes extra whitespaces

    return text


def clean_for_multi(text):
    return clean_preprocess(text, do_farasa_tokenization=None,
                            farasa=None,
                            use_farasapy=True, )


def verify_dataset(dataset, has_answers=True):
    assert len(dataset) == len({entry.pq_id for entry in dataset}), "There are some non unique pd_ids,please check"
    if has_answers:
        for entry in dataset:
            print_mismatched_answers(entry)
        assert all(len(entry.answers) > 0 for entry in dataset), "There are some empty answer lists, please check"
        assert all(answer.text.strip() for entry in dataset for answer in entry.answers), "There are some empty answer texts, please check"


def print_mismatched_answers(entry):
    for answer in entry.answers:
        extracted = entry.passage[answer.start_char:answer.start_char + len(answer.text)]
        if extracted != answer.text:
            answer_in_context = answer.text in entry.passage
            print(f"'{answer.text}', '{extracted}', {answer_in_context}")  # this answer text is not found the paragraph, it's too rare I will neglect


def save_multi_processed(train_save_path, has_answers=True):
    dataset = read_JSONL_file(train_save_path)
    for entry in dataset:
        entry.passage = clean_for_multi(entry.passage)
        entry.question = clean_for_multi(entry.question)
        for answer in entry.answers:
            answer.text = clean_for_multi(answer.text)
            answer.start_char = entry.passage.index(answer.text)
    processed_path = train_save_path.replace("train", "preprocessed_train") \
        .replace("dev", "preprocessed_dev") \
        .replace("test", "preprocessed_test")
    write_to_JSONL_file(dataset, processed_path)
    verify_dataset(dataset, has_answers=has_answers)


def save_downloaded_file(save_path, file_url):
    if not os.path.exists(save_path):
        train_data = requests.get(file_url)
        if train_data.status_code in (200,):
            with open(save_path, "wb") as file:
                file.write(train_data.content)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_qrcd2023()
    download_qrcd()
