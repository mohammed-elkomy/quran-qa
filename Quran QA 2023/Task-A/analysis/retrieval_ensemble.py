from collections import defaultdict

import pandas as pd


def retrieval_ensemble(model_runs_dfs, cutoff=1000):
    ensemble_agg = defaultdict(lambda: defaultdict(float))
    for model_run_df in model_runs_dfs:
        for _, row in model_run_df.iterrows():
            # ensemble_agg[row["qid"]][row["docid"]] += cutoff*10 - row["rank"]
            # ensemble_agg[row["qid"]][row["docid"]] += -row["rank"]
            ensemble_agg[row["qid"]][row["docid"]] += row["score"] * (1 / (row["rank"] + 1))
    ensemble_run_df = {
        "qid": [],
        "docid": [],
        "rank": [],
        "score": [],
    }
    for qid, doc_scores in ensemble_agg.items():
        top_docs = sorted(doc_scores.items(), key=lambda item: item[-1], reverse=True)
        min_doc_score = min([x for _, x in top_docs])
        for rank, (doc_id, doc_score) in enumerate(top_docs[:cutoff]):
            ensemble_run_df["qid"].append(qid)
            ensemble_run_df["docid"].append(doc_id)
            ensemble_run_df["rank"].append(rank)
            ensemble_run_df["score"].append(doc_score - min_doc_score)

    ensemble_run_df = pd.DataFrame(ensemble_run_df)
    ensemble_run_df["Q0"] = "Q0"
    return ensemble_run_df[["qid", "Q0", "docid", "rank", "score"]]


def optimal_retrieval_ensemble(model_results):
    per_query_aggregate = defaultdict(lambda: defaultdict(list))
    for result in model_results:
        for qid, q_results in result.items():
            for metric_name, metric_value in q_results.items():
                per_query_aggregate[qid][metric_name].append(metric_value)
    optimal_ensemble = defaultdict(dict)  # optimal ensemble obtains maximum performance per each query
    for qid, q_results in per_query_aggregate.items():
        for metric_name, metric_value in q_results.items():
            optimal_ensemble[qid][metric_name] = max(metric_value)

    return optimal_ensemble
