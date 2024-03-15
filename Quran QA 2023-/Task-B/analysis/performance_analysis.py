import glob
import io
import json
import os
import random
import re
import sys
import traceback
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluate import load
from joypy import joyplot
from tqdm import tqdm

from answer_list_post_processing.post_processing import post_process_answer_list
from data_scripts.loader_scripts.read_write_qrcd import read_JSONL_file
from metrics.QQA23_metric import get_ntop_submission, convert_to_metric_expected_format
from metrics.QQA23_TaskB_eval import load_jsonl

CUT_OFF = 10
NO_ANSWER_THRESHOLD = 0.8
QUICK = False
WRITE_FREQ = 20

ref_files = {ref_file: load_jsonl(ref_file) for ref_file in glob.glob("data/*.jsonl")}

FAITHFUL_LIST = [
    # "D1_my_in_leakage",
    "D2_my_in_no_leakage",
    "D3_my_ood_hard",
    # "D4_my_ood_easy",
]

METRIC_QQA23 = load("metrics/QQA23_metric.py")


def get_model_name_acronym(model_name):
    if "araelectra" in model_name:
        return "araelectra"
    elif "arabertv02" in model_name:
        return "arabertv02"
    elif "camelbert" in model_name:
        return "camelbert"
    else:
        return model_name


def get_loss_type(model_name):
    if "MAL" in model_name:
        return "MAL"
    elif "random" in model_name:
        return "random"
    else:
        return "first"


def get_preditions_from_byteIO(dump_file, dump_file_io):
    dump_file_io.write(dump_file)
    dump_file_io.seek(0)

    dump_data = joblib.load(dump_file_io)

    return dump_data


def evaluate_categories(submission_data, data_split_seed, scores_diff):
    ref_files_paths = [os.path.join("data", f"{filename}-processed-{data_split_seed}_dev.jsonl")
                       for filename in FAITHFUL_LIST]

    category_results = dict.fromkeys(FAITHFUL_LIST, 0)  # initialize by zero
    for category_ref_file in ref_files_paths:
        if not os.path.exists(category_ref_file):
            # if the file doesn't depend on seed, remove it
            category_ref_file = category_ref_file.replace(f"-{data_split_seed}", "")
        if category_ref_file in ref_files:
            subset_ids = [e["pq_id"] for e in ref_files[category_ref_file]]
            submission_data_for_subset = {k: v for k, v in submission_data.items() if k in subset_ids}
            scores_diff_for_subset = {k: v for k, v in scores_diff.items() if k in subset_ids}
            metrics = metric_evaluation_wrapper(ref_files[category_ref_file], submission_data_for_subset, scores_diff_for_subset)
            category_id = re.findall(r"D\d", category_ref_file)
            assert len(category_id)
            category_id = category_id[0]
            cate_name = [cate_name
                         for cate_name in FAITHFUL_LIST
                         if category_id in cate_name][0]

            category_results[cate_name] = metrics["official_pAP@10"]
        else:
            assert False

    return category_results


def metric_evaluation_wrapper(dataset_jsonl, ntop_predictions, score_diff, no_answer_threshold=NO_ANSWER_THRESHOLD):
    references_, predictions_ = convert_to_metric_expected_format(dataset_jsonl, ntop_predictions)

    return METRIC_QQA23.compute(predictions=predictions_,
                                references=references_,
                                score_diff=score_diff,
                                no_answer_threshold=no_answer_threshold,
                                cutoff=CUT_OFF)


def evaluate_with_post_processing(dataset_jsonl, submission_data, scores_diff):
    submission_data_post = post_process_answer_list(submission_data, dataset_jsonl)
    post_eval_results = metric_evaluation_wrapper(dataset_jsonl, submission_data_post, scores_diff)
    return post_eval_results


def extract_dump_data(dump_file, folder_name, do_post, faithful):
    with ZipFile(dump_file, "r") as zip_file:
        data_frame_row = {}

        run_data_file = [file.filename for file in zip_file.filelist if "combined" in file.filename]
        assert len(run_data_file) == 1
        run_data_file = run_data_file[0]

        run_data = json.loads(zip_file.read(run_data_file))
        assert run_data['eval_NoAns_total'] + run_data['eval_SingleAns_total'] + run_data['eval_MultiAns_total'] == run_data['eval_total']
        if run_data["best_model"]:
            model_name = run_data["best_model"].split("/")[0]
        else:
            model_name = os.path.split(dump_file)[-1].rsplit("-", maxsplit=2)[0]  # for train dev having no model
        ##########################################
        eval_raw_predictions, eval_scores_diff = get_submission_from_zip(zip_file, "eval")
        test_raw_predictions, test_scores_diff = get_submission_from_zip(zip_file, "predict")

        eval_submission_data = get_ntop_submission(eval_raw_predictions, CUT_OFF, no_answer_threshold=NO_ANSWER_THRESHOLD)  # truncates answers list

        if do_post:
            ref_file_name = run_data['validation_file'].replace("../../", "")
            dataset_jsonl = load_jsonl(ref_file_name)

            post_eval_results = evaluate_with_post_processing(dataset_jsonl, eval_submission_data, eval_scores_diff)
            for k, v in post_eval_results.items():
                data_frame_row[f"post_eval_" + k] = v

        if faithful:  # the original split has no categories to evaluate against
            data_split_seed = os.path.split(run_data["train_file"])[-1].replace("_train.jsonl", "").rsplit("-", maxsplit=1)[-1]
            cate_results = evaluate_categories(eval_submission_data, data_split_seed, eval_scores_diff)
            for cate_name, cate_pAP in cate_results.items():
                data_frame_row[f"{cate_name}_pAP"] = cate_pAP

        for k, v in run_data.items():
            if "eval" in k:
                data_frame_row[k] = v

        data_frame_row["model_name"] = get_model_name_acronym(model_name)
        data_frame_row["split_name"] = folder_name
        data_frame_row["epochs"] = run_data["epoch"]
        data_frame_row["seed"] = run_data["seed"]
        data_frame_row["lr"] = run_data["training_args"]["learning_rate"]
        data_frame_row["loss_type"] = get_loss_type(model_name)
        data_frame_row["subfolder"] = os.path.split(dump_file.replace(f"dumps/{folder_name}", ""))[0]
        data_frame_row["qa_pretrained"] = "fine-tuned" in model_name
        data_frame_row["dump_file"] = dump_file

    cache_data = {"eval_raw_predictions": eval_raw_predictions,
                  "test_raw_predictions": test_raw_predictions,
                  "eval_scores_diff": eval_scores_diff,
                  "test_scores_diff": test_scores_diff,}

    return data_frame_row, cache_data


def get_submission_from_zip(zip_file, split):
    eval_dump_file = [file.filename for file in zip_file.filelist if f"{split}-" in file.filename and file.filename.endswith(".dump")]
    assert len(eval_dump_file) == 1
    eval_dump_file = eval_dump_file[0]
    eval_dump_file = zip_file.read(eval_dump_file)
    with io.BytesIO() as eval_dump_file_io:
        model_dump = get_preditions_from_byteIO(eval_dump_file, eval_dump_file_io)
        raw_predictions = model_dump['formatted_predictions']
        scores_diff = model_dump['scores_diff_json']
    return raw_predictions, scores_diff


def collect_dumps_dataframe(folder_name, do_post, write_folder_cache, faithful=False):
    dump_path = f"artifacts/summary/{folder_name}.dump"
    print(f"Processing {dump_path}..")

    if os.path.exists(dump_path) or QUICK:
        df = joblib.load(dump_path)

        if QUICK:
            return pd.DataFrame(df)
    else:
        df = []

    joblib.dump(df, dump_path.replace(".dump", "-backup.dump"), compress=5)
    disk_dump_file = []
    update = False

    walk = list(os.walk(f"artifacts/dumps/{folder_name}/"))
    for root, folders, files in tqdm(walk):

        if len(folders) == 0:
            cache_path = os.path.join(root, "cache.dmp")
            folder_cache = get_cache_data(cache_path)
            for dump_file in tqdm(files):
                dump_file = os.path.join(root, dump_file)
                if dump_file.endswith(".zip"):
                    seen_dump_files = [e["dump_file"] for e in df]
                    if dump_file not in seen_dump_files or dump_file not in folder_cache:
                        data_frame_row, cache_data = extract_dump_data(dump_file, folder_name, do_post, faithful)
                        folder_cache[dump_file] = cache_data
                        df.append(data_frame_row)
                        update = True

                    if len(df) % WRITE_FREQ == 0 and update:
                        print("writing")
                        joblib.dump(df, dump_path, compress=5)
                        if write_folder_cache:
                            joblib.dump(folder_cache, cache_path, compress=5)

                    disk_dump_file.append(dump_file)
            if write_folder_cache:
                folder_cache = {k: v for k, v in folder_cache.items() if k in disk_dump_file}
                joblib.dump(folder_cache, cache_path, compress=5)

    filter_df = []
    for entry in df:
        if entry["dump_file"] in disk_dump_file and entry["dump_file"] not in [f_entry["dump_file"] for f_entry in filter_df]:
            filter_df.append(entry)

    df = filter_df
    assert sorted(disk_dump_file) == sorted([e["dump_file"] for e in df])  # same list
    joblib.dump(df, dump_path, compress=5)
    joblib.dump(df, dump_path.replace(".dump", "-backup.dump"), compress=5)

    return pd.DataFrame(df)


def get_cache_data(cache_path):
    if os.path.exists(cache_path):
        try:
            folder_cache = joblib.load(cache_path)
        except:
            print()
            folder_cache = {}
    else:
        folder_cache = {}
    return folder_cache


if __name__ == "__main__":
    test_df = collect_dumps_dataframe("merged", do_post=False, write_folder_cache=True,)
    original_df = collect_dumps_dataframe("original", do_post=True, write_folder_cache=True)
    faithful_df = collect_dumps_dataframe("faithful", do_post=True, write_folder_cache=False, faithful=True)


