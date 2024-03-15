from transformers import EvalPrediction




def compute_metrics(p: EvalPrediction, metric, no_answer_threshold, cutoff):
    """
    :param p: result of a model
    :param metric: metric from HF eval or dataset metric (same api)
    :param cutoff: max answers to be included
    :param no_answer_threshold: threshold for no-answer samples
    :return: metric results
    """
    # predictions = [{'prediction_text': '1999', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
    # references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]

    metric_results = metric.compute(predictions=p.predictions, references=p.label_ids, no_answer_threshold=no_answer_threshold, cutoff=cutoff)
    return metric_results
