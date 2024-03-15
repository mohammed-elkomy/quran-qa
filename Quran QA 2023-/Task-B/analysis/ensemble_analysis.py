import hashlib
import json
import os

import joblib

from tqdm import tqdm

from analysis.performance_analysis import CUT_OFF, metric_evaluation_wrapper
from answer_list_post_processing.post_processing import post_process_answer_list
from metrics.QQA23_TaskB_eval import load_jsonl
from ensemble.voting_ensemble import voting_ensemble, optimal_ensemble_pAP
from metrics.QQA23_metric import get_ntop_submission
from metrics.compute_score_qrcd import normalize_score_diff
from text_processing_helpers import verify_positional_correctness

NO_ANSWER_THRESHOLD = .8


def apply_cutoff(submission):
    for k in submission:
        submission[k] = submission[k][:CUT_OFF]
    return submission


def ensemble_for_directory(directory, is_eval, per_sample_performance=None):
    if is_eval:
        ref_file = load_jsonl("data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl")
        split = "eval"
    else:
        ref_file = load_jsonl("data/QQA23_TaskB_qrcd_v1.2_test_preprocessed.jsonl")
        split = "test"

    dmp_path = f"artifacts/summary/ensemble-{split}.dmp"
    ensemble_cache_dump = {}
    walk = list(os.walk(directory))
    for root, folders, files in tqdm(walk, total=len(walk)):
        if len(folders) == 0:
            cache_path = os.path.join(root, "cache.dmp")
            if os.path.exists(cache_path):
                dumps_dict = joblib.load(cache_path)
                num_models = len(dumps_dict)

                model_runs = [checkpoint[f"{split}_raw_predictions"] for checkpoint in dumps_dict.values()]
                model_diff_scores = [normalize_score_diff(checkpoint[f"{split}_scores_diff"]) for checkpoint in dumps_dict.values()]

                ensemble_outputs, ensemble_no_ans_scores = voting_ensemble(model_runs, model_diff_scores)

                for model_run in model_runs:
                    verify_positional_correctness(model_run, ref_file)
                verify_positional_correctness(ensemble_outputs, ref_file)

                ensemble_submission_data = get_ntop_submission(ensemble_outputs, cutoff=100, no_answer_threshold=None)  # truncates answers list
                ensemble_submission_data_post = post_process_answer_list(ensemble_submission_data, ref_file)
                experiment_name = root.replace("artifacts/dumps/", "").replace("/", "-")
                with open(f"artifacts/submissions/{experiment_name}-ensemble-post-{split}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(apply_cutoff(ensemble_submission_data_post)))
                run_code = get_run_code(f"{experiment_name}-ensemble-post-{split}")
                with open(f"artifacts/submissions/TCE_{run_code}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(apply_cutoff(ensemble_submission_data_post)))

                with open(f"artifacts/submissions/{experiment_name}-ensemble-{split}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(apply_cutoff(ensemble_submission_data)))

                run_code = get_run_code(f"{experiment_name}-ensemble-{split}")
                with open(f"artifacts/submissions/TCE_{run_code}.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(apply_cutoff(ensemble_submission_data)))

                if is_eval:
                    ensemble_submission_data = get_ntop_submission(ensemble_outputs, cutoff=100, )  # truncates answers list
                    per_sample_for_models = [per_sample for model_key, per_sample in per_sample_performance.items() if any([file_name in model_key for file_name in files])]
                    ensemble_results = metric_evaluation_wrapper(ref_file, ensemble_submission_data, ensemble_no_ans_scores, no_answer_threshold=NO_ANSWER_THRESHOLD)
                    post_ensemble_results = metric_evaluation_wrapper(ref_file, ensemble_submission_data_post, ensemble_no_ans_scores, no_answer_threshold=NO_ANSWER_THRESHOLD)
                    ensemble_cache_dump[root] = {
                        "raw_ensemble_outputs": ensemble_outputs,
                        "raw_eval_ensemble": ensemble_results,
                        "post_eval_ensemble": post_ensemble_results,
                        "optimal_ensemble_pAP@10": optimal_ensemble_pAP(per_sample_for_models),
                        "num_models": num_models,
                        "ensemble_submission_data": ensemble_submission_data,
                        "ensemble_submission_data_post": ensemble_submission_data_post,
                        "ensemble_no_ans_scores": ensemble_no_ans_scores,
                    }
                    joblib.dump(ensemble_cache_dump, dmp_path, compress=5)


def get_run_code(original):
    run_code = hashlib.sha1(original.encode()).hexdigest()[-9:]
    print("\n", original, "=>", run_code)
    return run_code


if __name__ == "__main__":
    per_sample_performance = {model_run["dump_file"]: model_run["eval_per_sample"]
                              for model_run in joblib.load("artifacts/summary/original.dump")}

    ensemble_for_directory("artifacts/dumps/merged", is_eval=False)  # test phase
    ensemble_for_directory("artifacts/dumps/original", is_eval=True, per_sample_performance=per_sample_performance)
