from collections import defaultdict

import numpy as np
import pandas as pd


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, per_query_raw, query_to_doc_preds, query_to_doc_ground):
    preds = {k: len(v) > 0 for k, v in query_to_doc_preds.items()}  # whether each query is predicted to have and answer or not
    map_raw = {q: v['map_cut_10'] for q, v in per_query_raw.items()}
    recip_raw = {q: v['recip_rank'] for q, v in per_query_raw.items()}
    qid_to_has_ans = {k: "-1" not in v for k, v in query_to_doc_ground.items()}
    na_probs = {k: -sum(v.values()) for k, v in query_to_doc_preds.items()}  # negatively proportional to the total doc scores
    min_lim, max_lim = min(na_probs.values()), max(na_probs.values())
    if min_lim != max_lim:
        na_probs = {k: (v - min_lim) / (max_lim - min_lim) for k, v in na_probs.items()}
    else:
        na_probs = {k: 0 for k, v in na_probs.items()}  # we have same scores we cant normalize
    best_map, map_thresh = find_best_thresh(preds, map_raw, na_probs, qid_to_has_ans)
    best_recip, recip_thresh = find_best_thresh(preds, recip_raw, na_probs, qid_to_has_ans)
    main_eval["best_map"] = best_map
    main_eval["best_map_thresh"] = map_thresh
    main_eval["best_recip"] = best_recip
    main_eval["best_recip_thresh"] = recip_thresh


def evaluate_subset(per_sample, subset_ids):
    avg_dict = defaultdict(float)
    for qid, scores in per_sample.items():
        if qid in subset_ids:
            for k, v in scores.items():
                avg_dict[k] += v
    for k in avg_dict:
        avg_dict[k] /= len(subset_ids)
    return dict(avg_dict)


def split_ids(df_qrels):
    data_dict = df_qrels.groupby("qid").apply(lambda x: list(x["docid"])).to_dict()
    no_answer = [k for k, v in data_dict.items() if v == ["-1"]]
    single_answer = [k for k, v in data_dict.items() if len(v) == 1 and k not in no_answer]
    multi_answer = [k for k, v in data_dict.items() if len(v) > 1]
    return no_answer, single_answer, multi_answer


def convert_to_dict(df, column1, column2, column3):
    '''Convert a dataframe to dictionary of dictionaries to match the TREC eval format
    column1: should be the query id column
    column2: should be the docid column
    column3: can be either the relevance column (in case of qrels) or the score column (in case of a run)
    '''
    grouped_dict = df.groupby(column1).apply(lambda x: x.set_index(column2)[column3].to_dict()).to_dict()
    # sample output:
    # qrel = {
    #     'q1': {
    #         'd1': 0,
    #         'd2': 1,
    #     },
    #     'q2': {
    #         'd2': 1,
    #         'd3': 1,
    #     },
    # }
    return grouped_dict


def apply_threshold(df_run, no_answer_threshold, reject_percent=0.15):
    query_to_doc_preds = convert_to_dict(df_run, column1='qid', column2='docid', column3='score')

    na_probs = {k: -sum(v.values()) for k, v in query_to_doc_preds.items()}  # negatively proportional to the total doc scores
    min_lim, max_lim = min(na_probs.values()), max(na_probs.values())
    na_probs = {k: (v - min_lim) / (max_lim - min_lim) for k, v in na_probs.items()}
    return_threshold = False
    if no_answer_threshold is None:
        no_answer_threshold = np.quantile(list(na_probs.values()), 1 - reject_percent)
        return_threshold = True
    assert 0 <= no_answer_threshold <= 1
    if no_answer_threshold == 0:  # for corner cases
        no_answer_threshold -= 1e-5
    elif no_answer_threshold == 1:
        no_answer_threshold += 1e-5
    no_answer_q = [k for k, v in na_probs.items() if v > no_answer_threshold]  # after thresholding, those queries are marked as unanswerable
    clipped_df = df_run[~df_run["qid"].isin(no_answer_q)]
    no_ans_df = pd.DataFrame({"qid": no_answer_q,
                              "Q0": "Q0",
                              "docid": "-1",
                              "rank": 1,
                              "score": 100,
                              "tag": f"thresh@{no_answer_threshold:0.2f}"
                              })
    if return_threshold:
        return pd.concat([clipped_df, no_ans_df]), no_answer_threshold
    else:
        return pd.concat([clipped_df, no_ans_df])


def apply_cutoff(df_run, cutoff):
    query_to_doc_preds = df_run.groupby('qid')

    dfs = []
    for _, df in query_to_doc_preds:
        dfs.append(df.sort_values("score", ascending=False).head(cutoff))

    return pd.concat(dfs)
