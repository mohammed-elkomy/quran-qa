import io
import os
from collections import defaultdict
from zipfile import ZipFile

import pandas as pd

from tqdm import tqdm

from analysis.retrieval_ensemble import retrieval_ensemble, optimal_retrieval_ensemble
from data_scripts import read_qrels_file, read_run_file
from metrics.Custom_TaskA_eval import evaluate_task_a, evaluation_after_thresh
from metrics.helpers import apply_threshold, apply_cutoff

CUT_OFF = 10


def read_from_zip(zip_file, file_to_read):
    with io.BytesIO() as eval_dump_file_io:
        eval_dump_file_io.write(zip_file.read(file_to_read))
        eval_dump_file_io.seek(0)
        df_run = pd.read_csv(eval_dump_file_io, sep="\t", names=["qid", "Q0", "docid", "rank", "score", "tag"])
        df_run["docid"] = df_run["docid"].astype(str)
        df_run["qid"] = df_run["qid"].astype(str)
        return df_run


def evaluate_steps(raw_run_df, no_answer_threshold):
    raw_split_results, _ = evaluate_task_a(df_qrels, raw_run_df)
    # evaluation_after_thresh(df_qrels, raw_run_df, 0)
    # evaluation_after_thresh(df_qrels, raw_run_df, 1)
    rets = apply_threshold(raw_run_df, no_answer_threshold=no_answer_threshold)
    if no_answer_threshold is None:
        raw_run_df, quantile_threshold = rets
    else:
        raw_run_df = rets
        quantile_threshold = None

    thre_run_df = apply_cutoff(raw_run_df, cutoff=CUT_OFF)
    thre_split_results, thre_per_sample_results = evaluate_task_a(df_qrels, thre_run_df)

    thre_split_results["overall"]["best_map"] = raw_split_results["overall"]["best_map"]
    thre_split_results["overall"]["best_map_thresh"] = raw_split_results["overall"]["best_map_thresh"]
    thre_split_results["overall"]["best_recip"] = raw_split_results["overall"]["best_recip"]
    thre_split_results["overall"]["best_recip_thresh"] = raw_split_results["overall"]["best_recip_thresh"]
    thre_split_results["overall"]["quantile threshold"] = quantile_threshold
    return thre_split_results, thre_per_sample_results


if __name__ == "__main__":
    all_results_df = []
    df_qrels = read_qrels_file("data/QQA23_TaskA_qrels_dev.gold")
    walk = list(os.walk("artifacts/dumps"))
    summary_dump = {}
    ensemble_runs = defaultdict(list)
    for root, folders, files in tqdm(walk):
        if len(folders) == 0:
            summary_dump[root] = {}
            for dump_file in files:
                dump_file = os.path.join(root, dump_file)
                if dump_file.endswith(".zip"):
                    with ZipFile(dump_file, "r") as zip_file:
                        eval_file = [f.filename for f in zip_file.filelist if "eval" in f.filename]
                        assert len(eval_file) == 1
                        run_df = read_from_zip(zip_file, file_to_read=eval_file[0])
                        ensemble_runs[root].append(run_df)
                        split_results, per_sample_results = evaluate_steps(run_df, None)
                        summary_dump[root][os.path.split(dump_file)[-1]] = split_results, per_sample_results

    ensemble_results = {}
    for model_dir, model_runs in ensemble_runs.items():
        ensemble_df = retrieval_ensemble(model_runs, cutoff=CUT_OFF)
        ensemble_split_results, ensemble_per_sample_results = evaluate_steps(ensemble_df, None)
        print(model_dir, ensemble_split_results["overall"]["best_map_thresh"])
        ensemble_results[model_dir] = ensemble_split_results, ensemble_per_sample_results

    ensemble_final_results = []
    full_summary = []
    for model_dir, model_results in summary_dump.items():
        model_results_df = pd.concat([pd.json_normalize(result[0]) for result in model_results.values()])
        model_results_df["file_name"] = model_results.keys()
        model_results_df["model_root"] = model_dir

        model_results_df.to_excel(os.path.join("artifacts/summary/", os.path.split(model_dir)[-1]) + ".xlsx")
        all_results_df.append(model_results_df)
        full_summary.append(model_results_df)

        optimal_ensemble = optimal_retrieval_ensemble([result[1] for result in model_results.values()])

        optimal_ensemble_map_cut_10 = sum([v["map_cut_10"] for v in optimal_ensemble.values()]) / len(optimal_ensemble)

        ensemble_final_results.append({
            "model_dir": model_dir,
            "original_avg_performance (map_cut_10)": model_results_df["overall.map_cut_10"].mean(),
            "original_min_performance (map_cut_10)": model_results_df["overall.map_cut_10"].min(),
            "actual_ensemble_result (map_cut_10)": ensemble_results[model_dir][0]["overall"]["map_cut_10"],
            "best_original_avg_performance (best_map)": model_results_df["overall.best_map"].mean(),
            "best_actual_ensemble_result (best_map)": ensemble_results[model_dir][0]["overall"]["best_map"],
            "optimal_ensemble (map_cut_10)": optimal_ensemble_map_cut_10,

            "original_avg_performance (single map_cut_10 )": model_results_df["single_answer.map_cut_10"].mean(),
            "original_avg_performance (multi map_cut_10)": model_results_df["multi_answer.map_cut_10"].mean(),
            "original_avg_performance (no answer map_cut_10)": model_results_df["no_answer.map_cut_10"].mean(),

            "original_avg_performance (recip_rank)": model_results_df["overall.recip_rank"].mean(),
            "best_original_avg_performance (best_recip_rank)": model_results_df["overall.best_recip"].mean(),
            "actual_ensemble_result (recip_rank)": ensemble_results[model_dir][0]["overall"]["recip_rank"],
            "best_actual_ensemble_result (best_recip_rank)": ensemble_results[model_dir][0]["overall"]["best_recip"],

            "original_avg_performance (recall_10)": model_results_df["overall.recall_10"].mean(),
            "original_avg_performance (recall_100)": model_results_df["overall.recall_100"].mean(),

            "num_models": len(model_results)
        })

    performance_df = pd.DataFrame(ensemble_final_results)
    performance_df.to_excel(os.path.join("artifacts/summary/", f"ensemble_performance_df.{CUT_OFF}.xlsx"))

    pd.concat(all_results_df).to_excel(f"artifacts/summary/all_models_results.{CUT_OFF}.xlsx")
