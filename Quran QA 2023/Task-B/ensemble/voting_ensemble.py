from collections import defaultdict, Counter

import numpy as np
from scipy.special import softmax


def softmax_t(x, temperature=.1):
    x = np.array(x)
    return softmax(x / temperature)


def update_scores(predictions, score_fn):
    for prediction in predictions:
        rankedlist_len = len(prediction["answers"]["score"])
        if score_fn == "reciprocal":
            prediction["answers"]["score"] = [1 / rank for rank in range(1, rankedlist_len)]
        elif score_fn == "linear":
            prediction["answers"]["score"] = [(rankedlist_len - rank) / rankedlist_len for rank in range(rankedlist_len)]
        elif score_fn == "softmax":
            # keep it
            assert np.isclose(sum(prediction["answers"]["score"]), 1)
    return predictions


def voting_ensemble_per_sample(per_sample_data):
    aggregated = defaultdict(float)
    aggregated_no_ans_prop = 0
    for sample_data_model in per_sample_data:
        for s, e, text, score in zip(sample_data_model["answers"]["start_token_index"],
                                     sample_data_model["answers"]["end_token_index"],
                                     sample_data_model["answers"]["text"],
                                     sample_data_model["answers"]["score"]
                                     ):
            answer_span = s, e, text
            aggregated[answer_span] += score
        aggregated_no_ans_prop += sample_data_model["no_answer_probability"]

    sorted_predictions = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

    voting_result = {
        "start_token_index": [k[0] for k, v in sorted_predictions],
        "end_token_index": [k[1] for k, v in sorted_predictions],
        "text": [k[2] for k, v in sorted_predictions],
        "score": [v for k, v in sorted_predictions],
        "rank": list(range(1, len(sorted_predictions) + 1)),
    }
    num_predictors = len(per_sample_data)
    return voting_result, aggregated_no_ans_prop / num_predictors


def voting_ensemble(model_runs, model_diff_scores, update=True, score_fn="reciprocal"):
    per_sample = defaultdict(list)
    for model_run in model_runs:
        if update:
            model_run = update_scores(model_run, score_fn=score_fn)
        for sample_data in model_run:
            per_sample[sample_data["id"]].append(sample_data)

    voting_ensemble_results = []
    for pq_id, checkpoints_answer_list in per_sample.items():
        answers_list, no_ans_prop = voting_ensemble_per_sample(checkpoints_answer_list)
        assert 0 <= no_ans_prop <= 1
        voting_ensemble_results.append({
            "id": pq_id,
            "answers": answers_list,
            "no_answer_probability": no_ans_prop
        })

    ensemble_no_ans_scores = defaultdict(float)
    for model_diff_score in model_diff_scores:
        for sample_id, sample_score in model_diff_score.items():
            ensemble_no_ans_scores[sample_id] += sample_score
    return voting_ensemble_results, ensemble_no_ans_scores


def optimal_ensemble_pAP(models_per_sample):
    per_query = defaultdict(list)
    for model_per_sample in models_per_sample:
        for qid, score in model_per_sample.items():
            per_query[qid].append(score)

    per_query = {qid: max(scores) for qid, scores in per_query.items()}
    return np.mean(list(per_query.values()))
