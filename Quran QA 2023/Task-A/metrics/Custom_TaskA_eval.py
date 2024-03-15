import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytrec_eval

sys.path.append(os.getcwd())  # for relative imports
sys.path.append("..")  # for relative imports
sys.path.append("../..")  # for relative imports

from data_scripts import read_qrels_file, read_run_file
from metrics.helpers import evaluate_subset, split_ids, find_all_best_thresh, apply_threshold, convert_to_dict

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

METRICS = ["map_cut_10", "recip_rank", "recall", "P"]


def get_metric_list(results_dict, metric):
    """ Extract the values from the result dictionary and put them in a list
    """
    values_list = [inner_dict[metric] for inner_dict in results_dict.values()]
    return values_list


def evaluate_zero_answer_questions(zero_answer_question_ids, df_run):
    """ Evaluate the performance for the no-answer questions
    Simply, for each no-answer question in the qrels file:
    If the run has only one retrieved document and this document has the id of -1, 
     then the system receives a full score 
    Otherwise: the run it will receive a zero score for that question
    zero_answer_question_ids: the ids of the no-answer questions
    df_run: the run of the no-answer questions, which needs to be evaluated
    """

    scores = []
    run_question_ids = df_run["qid"].values

    for qid in zero_answer_question_ids:
        if qid not in run_question_ids:
            # if this question does not exist in the run file, give it a zero credit
            scores.append(0)

        # select the rows where the docid equals to the current docid
        retrieved_doc_ids = df_run.loc[df_run["qid"] == qid, "docid"].values

        if len(retrieved_doc_ids) == 1 and retrieved_doc_ids[0] == "-1":
            # if there is only one retrieved document and this document has the id of -1,
            # then the system receives a full score
            scores.append(1)
        else:
            # otherwise: it will receive a zero score
            scores.append(0)

    return scores, dict(zip(zero_answer_question_ids, scores))


def evaluate_normal_questions(df_run, df_qrels):
    """ Evaluate the performance for the normal questions
    df_run: the run dataframe of the normal questions
    df_qrels: the qrels dataframe of the normal questions
    The evaluation is performed using pytrec_eval tool (common tool for evaluating IR sysetems in python)
    """
    # convert the qrels to a dictionary to match the pytrec_eval format
    qrels_dict = convert_to_dict(df_qrels,
                                 column1="qid",
                                 column2="docid",
                                 column3="relevance")
    # convert the run into a dictionary  to match the pytrec_eval format
    run_dict = convert_to_dict(df_run,
                               column1="qid",
                               column2="docid",
                               column3="score")

    # initialize the evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, METRICS)
    # do the evaluation for each metric
    eval_res = evaluator.evaluate(run_dict)

    # extract the result values and put them in lists
    results = {}
    for metric in METRICS:
        metric_keys = [k for k in list(eval_res.values())[0].keys() if metric in k]
        for metric_key in metric_keys:
            metric_scores_list = get_metric_list(results_dict=eval_res, metric=metric_key)
            results.update({metric_key: metric_scores_list})

    return results, eval_res


def evaluation_after_thresh(df_qrels, df_run, threshold):
    # assert 0 <= threshold <= 1
    rets = apply_threshold(df_run, threshold)
    if threshold is None:
        thresh_df_run, quantile_threshold = rets
    else:
        thresh_df_run = rets

    per_split_result, per_query = evaluate_task_a(df_qrels, thresh_df_run)
    per_split_result["overall"].pop("best_map")
    per_split_result["overall"].pop("best_map_thresh")
    per_split_result["overall"].pop("best_recip")
    per_split_result["overall"].pop("best_recip_thresh")
    return per_split_result, per_query


def get_optimal_map(query_to_doc_ground, query_to_doc_preds):
    query_to_docs = {k: sorted(v.items(), key=lambda item: item[1], reverse=True)[:10] for k, v in query_to_doc_preds.items()}
    query_to_docs = {k: {vv[0] for vv in v} for k, v in query_to_docs.items()}
    res = {k: (len(query_to_docs[k].intersection(v)) / len(v)) for k, v in query_to_doc_ground.items()}
    return sum(res.values()) / len(res)  # same as recall@10 but a sanity check


def evaluate_task_a(df_qrels, df_run, output_file=None):
    query_to_doc_ground = convert_to_dict(df_qrels, column1="qid", column2="docid", column3="relevance")
    query_to_doc_preds = convert_to_dict(df_run, column1="qid", column2="docid", column3="score")
    opt_map_cut_10 = get_optimal_map(query_to_doc_ground, query_to_doc_preds)
    max_pred_per_q = max([len(e) for e in query_to_doc_preds.values()])
    min_pred_per_q = min([len(e) for e in query_to_doc_preds.values()])
    # select the zero answer questions from the qrel file
    zero_answer_question_ids = df_qrels.loc[df_qrels["docid"] == "-1", "qid"].values
    # get the qrels of the normal questions
    df_qrels_normal_questions = df_qrels.loc[~df_qrels["qid"].isin(zero_answer_question_ids)]
    # divide the run into two dataframes, one contains the zero answer question
    df_zero_answer_questions = df_run[df_run["qid"].isin(zero_answer_question_ids)]
    # and the other contain the normal questions
    df_normal_questions = df_run[~df_run["qid"].isin(zero_answer_question_ids)]
    zero_answer_question_scores, zero_per_sample = evaluate_zero_answer_questions(zero_answer_question_ids, df_zero_answer_questions)
    normal_question_scores, normal_per_sample = evaluate_normal_questions(df_normal_questions, df_qrels_normal_questions)
    metric_keys_for_sample = list(list(normal_per_sample.values())[0].keys())
    zero_per_sample = {k: dict.fromkeys(metric_keys_for_sample, v) for k, v in zero_per_sample.items()}
    assert normal_per_sample.keys().isdisjoint(zero_per_sample.keys())
    per_query = {}
    per_query.update(zero_per_sample)
    per_query.update(normal_per_sample)
    overall_results = {}
    for metric in METRICS:
        metric_keys = [k for k in normal_question_scores.keys() if metric in k]
        for metric_key in metric_keys:
            metric_score_list = normal_question_scores[metric_key]
            metric_score_list.extend(zero_answer_question_scores)
            metric_score = np.mean(metric_score_list)
            overall_results.update({metric_key: metric_score})
    df = pd.DataFrame([overall_results])
    if output_file:
        df.to_csv(output_file, sep="\t", index=False)
        logger.info(f"Saved results to file: {output_file}")
    else:
        # print(df.to_string(index=False))
        pass
    no_answer, single_answer, multi_answer = split_ids(df_qrels)
    per_split_result = {
        "overall": overall_results,
        "single_answer": evaluate_subset(per_query, single_answer),
        "multi_answer": evaluate_subset(per_query, multi_answer),
        "no_answer": evaluate_subset(per_query, no_answer),

    }
    overall2 = evaluate_subset(per_query, set(df_qrels["qid"]))
    assert all([v - overall2[k] < 1e-5 for k, v in per_split_result["overall"].items()])
    per_split_result["min_preds_per_q"] = min_pred_per_q
    per_split_result["max_preds_per_q"] = max_pred_per_q

    find_all_best_thresh(per_split_result["overall"], per_query, query_to_doc_preds, query_to_doc_ground)
    per_split_result["overall"]["opt_map_cut_10"] = opt_map_cut_10
    return per_split_result, per_query


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", required=True,
                        help="run file with predicted scores from your model.\
                        Format: qid Q0 docid rank score tag")
    parser.add_argument("--qrels", "-q", required=True,
                        help="QRELS file with gold labels. Format: qid 0 docid relevance")
    parser.add_argument("--output", "-o",
                        help="Output file with metrics.\
                        If not specified, prints output in stdout.")
    return parser.parse_args()


def main(args):
    output_file = args.output
    qrels_file = args.qrels
    run_file = args.run
    # format_check_passed = qqa23_sc.check_run(run_file)
    # if not format_check_passed:
    #     return

    df_qrels = read_qrels_file(qrels_file)
    df_run = read_run_file(run_file)
    print(evaluation_after_thresh(df_qrels, df_run, 0.4))
    return evaluate_task_a(df_qrels, df_run, output_file)


if __name__ == "__main__":
    args = parse_args()

    # args.qrels = "/home/mohammed_elkomy/Downloads/dev-qrel.tsv"
    # args.run = "/home/mohammed_elkomy/Downloads/dev.rank2.tsv"

    # args.qrels = "/home/mohammed_elkomy/Downloads/dev-qrel-test.tsv"
    # args.run = "/home/mohammed_elkomy/Downloads/dev.rank2-test.tsv"

    #
    # args.qrels = "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/QQA23_TaskA_qrels_dev.gold"
    # args.run = "/home/mohammed_elkomy/Downloads/bigIR_BM25dev.tsv"
    # args.run = "/home/mohammed_elkomy/Downloads/dev.rank.tsv"
    # args.run = "/home/mohammed_elkomy/Downloads/dev.rank (1).tsv"

    results = main(args)
    print(results[0])
