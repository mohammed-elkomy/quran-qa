import os
from random import choice
from zipfile import ZipFile

import joblib
import numpy as np
from matplotlib import pyplot as plt

from analysis.performance_analysis import CUT_OFF, metric_evaluation_wrapper, get_submission_from_zip
from metrics.QQA23_TaskB_eval import load_jsonl
from metrics.QQA23_metric import get_ntop_submission

ref_file = load_jsonl("data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl")
ensemble_data = joblib.load("artifacts/summary/ensemble-eval.dmp")


def make_figure(raw_preds, no_ans_scores):
    x_vals = []
    y_vals = []
    for thresh in np.arange(0, 1, .05):
        submission_data = get_ntop_submission(raw_preds, CUT_OFF, no_answer_threshold=thresh)
        eval_results = metric_evaluation_wrapper(ref_file, submission_data, no_ans_scores)
        x_vals.append(thresh)
        y_vals.append(eval_results['official_pAP@10'])
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.xlabel('Threshold')
    plt.ylabel('MAP performance', )
    plt.title('Effect of thresholding', )


for root, folders, files in os.walk("artifacts/dumps/original/"):
    if len(folders) == 0:
        dump_file = choice([f for f in files if "zip" in f])
        dump_file = os.path.join(root, dump_file)
        with ZipFile(dump_file, "r") as zip_file:
            eval_raw_predictions, eval_scores_diff = get_submission_from_zip(zip_file, "eval")
            make_figure(eval_raw_predictions, eval_scores_diff)
            dump_file = dump_file.replace("/", "-")
            plt.savefig(f"artifacts/curves/{dump_file}.png")
            print(f"artifacts/curves/{dump_file}.png")
            plt.close()

for k, ensemble_instance in ensemble_data.items():
    if "pipelined" in k:
        make_figure(raw_preds=ensemble_instance["raw_ensemble_outputs"], no_ans_scores=ensemble_instance["ensemble_no_ans_scores"])
        # Displaying the chart
        k = k.replace("/", "-")
        plt.savefig(f"artifacts/{k}.png")
        print(f"artifacts/curves/{k}.png")
        plt.close()
