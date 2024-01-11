import glob
import json
import os
import random
from collections import defaultdict
from random import shuffle

import joblib
import numpy as np
from scipy.special import softmax

from data.qrcd_eval import load_jsonl, evaluate
from data.qrcd_format_checker import check_submission
from post_processing import smart_pRR_pruning

######################################
# how to use this script:
# this script works by finding all checkpoint files (ending with .dump) to combine them into an ensemble
# the inputs are dump files in results/eval and results/test
# the outputs will be saved to results/eval and results/test with json files named as the ensemble_original.json, ensemble_keep.json and ensemble_remove.json
# you may use the evaluation script to evaluate any of those files
######################################
max_answer_length = 35  # in tokens
DO_SOFTMAX = True


def softmax_t(x, temperature=.1):
    x = np.array(x)
    return softmax(x / temperature)


def check_order(first, second):
    return all(
        value1 == value2
        for value1, value2 in zip(first, second)
    )


def softmax_scores(predictions):
    for pq in predictions:
        normalized_scores = softmax_t([answer["score"] for answer in predictions[pq]])
        for answer, normalized_score in zip(predictions[pq], normalized_scores):
            answer["score"] = normalized_score

        # verify the scores after softmax
        scores = [answer["score"] for answer in predictions[pq]]
        ranks = [answer["rank"] for answer in predictions[pq]]
        assert check_order(sorted(scores)[::-1], scores)
        assert check_order(sorted(ranks), ranks)
    return predictions


def prepare_voting_ensemble(eval_dataset):
    return {entry["pq_id"]: {"candidate_answers": defaultdict(float)} for entry in eval_dataset}


def get_sub_file_from_pickle(pickle_path):
    data = joblib.load(pickle_path)
    top_k_predictions = data["top_k_predictions"]

    formatted_predictions = []
    for pred_id, top_k_prediction in top_k_predictions.items():
        answers_predictions = {"text": [k_th_prediction["text"] for k_th_prediction in top_k_prediction],
                               "score": [k_th_prediction["probability"] for k_th_prediction in top_k_prediction]}

        formatted_predictions.append({"id": pred_id, "answers": answers_predictions})

    ntop_predictions = {}
    for prediction in formatted_predictions:
        # collecting top k answers for this pq_id
        answers = [
            {"answer": text, "score": score, "rank": rank}
            for rank, (text, score) in enumerate(
                zip(prediction["answers"]["text"], prediction["answers"]["score"]), start=1
            )
        ]

        ntop_predictions[prediction["id"]] = answers[:20]  # 20 answers will be used for the ensemble

    file_name = f"tmp/tce_{random.randint(10, 10000000)}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(ntop_predictions))

    return file_name


def add_models_candidate_answers(dump_files, eval_ensemble):
    for dump_file in dump_files:
        run_file = get_sub_file_from_pickle(dump_file)  # the run_file is saved to tmp
        with open(run_file, 'r', encoding='utf-8') as run_file:
            ntop_predictions = json.load(run_file)

        if DO_SOFTMAX:
            ntop_predictions = softmax_scores(ntop_predictions)
        for pq_id, answers in ntop_predictions.items():
            answers_ensemble = eval_ensemble[pq_id]
            candidate_answers = answers_ensemble["candidate_answers"]
            for answer in answers:
                candidate_answers[answer["answer"]] += answer["score"]
                # candidate_answers[answer["answer"]] += 1 / answer["rank"]


def get_predictions_from_voting_ensemble(eval_ensemble):
    submission = {}
    for pq_id, pq_data in eval_ensemble.items():
        candidate_answers = pq_data["candidate_answers"]
        answers_for_pq_id = [{
            "score": score,
            "answer": candidate_answer
        } for candidate_answer, score in candidate_answers.items()]
        predictions = sorted(answers_for_pq_id, key=lambda x: x["score"], reverse=True)
        for rank, prediction in enumerate(predictions, start=1):
            prediction["rank"] = rank
        submission[pq_id] = predictions

    file_name = f"tmp/tce_{random.randint(10, 10000000)}.json"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(submission))

    return file_name


def truncate_to_5(answers_list):
    return {key: value[:5] for key, value in answers_list.items()}


def write_submission(file_name, submission):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(submission))
    print(f"saved {file_name}, you may use the official quranqa22_eval.py script to eval")


if __name__ == "__main__":
    print("ENSEMBLE FOR EVAL")
    dataset_jsonl = load_jsonl("data/qrcd/qrcd_v1.1_dev.jsonl")
    eval_ensemble = prepare_voting_ensemble(dataset_jsonl)

    ensemble_dump_files = glob.glob(os.path.join("post_processing/results", "eval/**/*.dump"))
    print("models for eval ensemble")
    print(ensemble_dump_files, "num of models", len(ensemble_dump_files))
    add_models_candidate_answers(ensemble_dump_files, eval_ensemble)  # this does the voting for each model into the answers voting list

    print("=" * 50)

    ensemble_original_submission_file = get_predictions_from_voting_ensemble(eval_ensemble)
    if check_submission(ensemble_original_submission_file) is False:
        print("Please review the above warning(s) or error message(s) related to this run file.")

    with open(ensemble_original_submission_file, 'r', encoding='utf-8') as ensemble_original_submission_file:
        ntop_predictions = json.load(ensemble_original_submission_file)

    # original
    original_eval_results = evaluate(dataset_jsonl, truncate_to_5(ntop_predictions))

    # keep
    ntop_predictions_keep = smart_pRR_pruning(ntop_predictions, dataset_jsonl)
    ntop_predictions_keep = truncate_to_5(ntop_predictions_keep)
    keep_uninformative_eval_results = evaluate(dataset_jsonl, ntop_predictions_keep)

    # reject
    ntop_predictions_reject = smart_pRR_pruning(ntop_predictions, dataset_jsonl, reject_answers=True)
    ntop_predictions_reject = truncate_to_5(ntop_predictions_reject)
    reject_uninformative_eval_results = evaluate(dataset_jsonl, ntop_predictions_reject)

    print("original results, without post-processing")
    write_submission("post_processing/results/eval/ensemble_original.json", truncate_to_5(ntop_predictions))
    print(original_eval_results)

    print("post-processing with uninformative answers kept")
    write_submission("post_processing/results/eval/ensemble_keep.json", ntop_predictions_keep)
    print(keep_uninformative_eval_results)

    print("post-processing with uninformative answers removed")
    write_submission("post_processing/results/eval/ensemble_remove.json", ntop_predictions_reject)
    print(reject_uninformative_eval_results)

    print("=" * 50)
    print("=" * 50)
    print("=" * 50)
    print("ENSEMBLE FOR TEST")
    dataset_jsonl = load_jsonl("data/qrcd/qrcd_v1.1_test_noAnswers.jsonl")
    test_ensemble = prepare_voting_ensemble(dataset_jsonl)

    ensemble_dump_files = glob.glob(os.path.join("post_processing/results", "test/**/*.dump"))  # test phase
    print("models for test ensemble")
    print(ensemble_dump_files, "num of models", len(ensemble_dump_files))
    add_models_candidate_answers(ensemble_dump_files, test_ensemble)  # this does the voting for each model into the answers voting list
    ensemble_original_submission_file = get_predictions_from_voting_ensemble(test_ensemble)
    if check_submission(ensemble_original_submission_file) is False:
        print("Please review the above warning(s) or error message(s) related to this run file.")

    with open(ensemble_original_submission_file, 'r', encoding='utf-8') as ensemble_original_submission_file:
        ntop_predictions = json.load(ensemble_original_submission_file)

    # keep
    ntop_predictions_keep = smart_pRR_pruning(ntop_predictions, dataset_jsonl)
    ntop_predictions_keep = truncate_to_5(ntop_predictions_keep)

    # reject
    ntop_predictions_reject = smart_pRR_pruning(ntop_predictions, dataset_jsonl, reject_answers=True)
    ntop_predictions_reject = truncate_to_5(ntop_predictions_reject)

    print("original results, without post-processing")
    write_submission("post_processing/results/test/ensemble_original.json", truncate_to_5(ntop_predictions))

    print("post-processing with uninformative answers kept")
    write_submission("post_processing/results/test/ensemble_keep.json", ntop_predictions_keep)

    print("post-processing with uninformative answers removed")
    write_submission("post_processing/results/test/ensemble_remove.json", ntop_predictions_reject)
