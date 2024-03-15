import collections
import logging
from copy import deepcopy
from itertools import product
from typing import Optional, Tuple

import joblib
import numpy as np

from text_processing_helpers import get_char_id_to_token_id_map

logger = logging.getLogger(__name__)


def decode_model_logits(
        unprocessed_dataset,
        processed_dataset,
        predictions: Tuple[np.ndarray, np.ndarray],
        n_best_size: int = 20,
        pairwise_decoder: bool = True,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        log_level: Optional[int] = logging.CRITICAL,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    @param log_level: (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
    ``logging`` log level (e.g., ``logging.WARNING``)

    @param prefix: (:obj:`str`, `optional`):
    If provided, the dictionaries mentioned above are saved with `prefix` added to their names.

    @param output_dir:  (:obj:`str`, `optional`):
    If provided, the dictionaries of predictions, n_best predictions (with their scores and logits)
    and the dictionary of the scores differences between best and null answers, are saved in `output_dir`.

    @param null_score_diff_threshold: (:obj:`float`, `optional`, defaults to 0):
    The threshold used to select the null answer: if the best answer has a score that is less than the score of
    the null answer minus this threshold, the null answer is selected for this example (note that the score of
    the null answer for an example giving several features is the minimum of the scores for the null answer on
    each feature: all features must be aligned on the fact they `want` to predict a null answer).

    @param max_answer_length: (:obj:`int`, `optional`, defaults to 30):
    The maximum length of an answer that can be generated. This is needed because the start and end predictions
    are not conditioned on one another.

    @param n_best_size:  (:obj:`int`, `optional`, defaults to 20):
    The total number of n-best predictions to generate when looking for an answer.

    @param predictions: (:obj:`Tuple[np.ndarray, np.ndarray]`):
    The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
    first dimension must match the number of elements of :obj:`features`.

    @param unprocessed_dataset: The non-preprocessed dataset (see the main script for more information).
    @param processed_dataset:  The processed dataset (see the main script for more information).
    @param pairwise_decoder: A boolean whether to use pairwise decoding or linear decoding, check training_args.py file for details
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")

    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(processed_dataset):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(processed_dataset)} features.")

    features_per_example = get_features_map(processed_dataset, unprocessed_dataset)

    # The dictionaries we have to fill.
    all_ntop_predictions = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(unprocessed_dataset)} example predictions split into {len(processed_dataset)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(unprocessed_dataset):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]  # if the passage is longer than the max_passage_length, feature_indices will be a list of features

        prelim_predictions, min_null_prediction = get_all_admissible_answers_for_sample(all_start_logits, all_end_logits,
                                                                                        processed_dataset,
                                                                                        feature_indices,
                                                                                        max_answer_length,
                                                                                        n_best_size, pairwise_decoder, )
        # prelim_predictions is a list of {"offsets": ,"score": ,"start_logit": ,"end_logit":} dictionaries

        # Add the minimum null prediction
        min_null_prediction["score"] -= null_score_diff_threshold  # the margin used to threshold non-empty answers
        prelim_predictions.append(min_null_prediction)
        null_score = min_null_prediction["score"] + null_score_diff_threshold  # this margin is used for comparison later, check update_ntop_predictions

        # Only keep the best `n_best_size` predictions.
        sorted_predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if all(p["offsets"] != (0, 0) for p in sorted_predictions):
            sorted_predictions.append(min_null_prediction)

        char_to_token = get_char_id_to_token_id_map(example["context"])

        sorted_predictions = extract_answer_text(example, sorted_predictions)  # adds a key "text" for each prediction
        sorted_predictions = get_probabilities_from_scores(sorted_predictions)  # adds a key "probability" for each prediction
        sorted_predictions = get_token_indexes(sorted_predictions, example, char_to_token)

        update_ntop_predictions(all_ntop_predictions,
                                example["id"],
                                sorted_predictions,
                                null_score,
                                null_score_diff_threshold,
                                scores_diff_json,
                                )  # all_ntop_predictions is updated for this "example"

    # try:
    #     assert all([all(["offsets" in vv for vv in v]) for v in all_ntop_predictions.values()])
    # except:
    #     joblib.dump(
    #         [
    #             unprocessed_dataset,
    #             processed_dataset,
    #             predictions,
    #             n_best_size,
    #             pairwise_decoder,
    #             max_answer_length,
    #             null_score_diff_threshold,
    #             output_dir,
    #             prefix,
    #             log_level,
    #         ],
    #         "big issue.dmp"
    #     )

    return all_ntop_predictions, scores_diff_json


def update_ntop_predictions(all_ntop_predictions, example_id, predictions, null_score, null_score_diff_threshold, scores_diff_json, ):
    """
    @param all_ntop_predictions: a dictionary to hold all ntop predictions (updated in this function)
    @param example_id: sample data
    @param predictions: sorted predictions
    @param null_score:
    @param null_score_diff_threshold:
    @param predictions:  predictions are sorted by score
    @param scores_diff_json:

    @return:
    """
    if 1 < len([e["text"] for e in predictions if not e["text"].strip()]) < len(predictions):
        joblib.dump(predictions, "many empty.dmp", compress=5)
        print("-----------\nMANY EMPTY\n-----------")

    i = 0
    while i < len(predictions) - 1 and predictions[i]["text"] == "":
        i += 1
    best_non_null_pred = predictions[i]

    # Then we compare to the null prediction using the threshold.
    score_diff = null_score - (best_non_null_pred["start_logit"] + best_non_null_pred["end_logit"])
    scores_diff_json[example_id] = float(score_diff)  # To be JSON-serializable.
    if null_score_diff_threshold < score_diff:  # the model is confident it's unanswerable, null_score is much greater  best_non_null
        if predictions[0]["text"] != "":
            joblib.dump(predictions, "issue.dmp", compress=5)
        assert predictions[0]["text"] == ""

    # Make `predictions` JSON-serializable by casting np.float back to float.
    all_ntop_predictions[example_id] = [
        {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
        for pred in predictions
    ]

    return all_ntop_predictions


def get_probabilities_from_scores(predictions):
    # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure.
    if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
        predictions.insert(0, {"text": "", "offsets": (0, 0), "start_logit": 0.0, "end_logit": 0.0, "score": 0.0, "exhausted_model": deepcopy(predictions)})

    # Compute the softmax of all scores (we do it with numpy to stay independent of torch/tf in this file, using the LogSumExp trick).
    scores = np.array([pred.pop("score") for pred in predictions])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    # Include the probabilities in our predictions.
    for prob, pred in zip(probs, predictions):
        pred["probability"] = prob

    return predictions


def extract_answer_text(example, predictions):
    # Use the offsets to gather the answer text in the original context.
    context = example["context"]
    for pred in predictions:
        offsets = pred["offsets"]
        pred["text"] = context[offsets[0]: offsets[1]]
    return predictions


def get_token_indexes(predictions, example, char_to_token):
    for pred in predictions:
        if pred["offsets"] == (0, 0):
            start_off, end_off = None, None
        else:
            start_off = char_to_token[pred["offsets"][0]]
            end_off = char_to_token[pred["offsets"][1] - 1]
            extracted = " ".join(example["context"].split()[start_off:end_off + 1])
            assert pred["text"] in extracted, f"{pred['text']}\n\n{extracted}"  # TODO remove this later
            pred["text"] = extracted  # sometimes the text predicted by the model has broken words, we fix it here

        pred["start_token_index"] = start_off
        pred["end_token_index"] = end_off

    return predictions


def get_all_admissible_answers_for_sample(all_start_logits, all_end_logits, processed_dataset, feature_indices, max_answer_length, n_best_size, pairwise_decoder, ):
    min_null_prediction = None
    prelim_predictions = []
    # Looping through all the features associated to the current example.
    for feature_index in feature_indices:
        # We grab the predictions of the model for this feature.
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        # This is what will allow us to map some the positions in our logits to span of texts in the original context.
        offset_mapping = processed_dataset[feature_index]["offset_mapping"]
        # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
        # available in the current feature.
        token_is_max_context = processed_dataset[feature_index].get("token_is_max_context", None)

        # Update minimum null prediction.
        feature_null_score = start_logits[0] + end_logits[0]
        if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
            min_null_prediction = {
                "offsets": (0, 0),
                "score": feature_null_score,
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
            }

        admissible_answers = get_admissible_answers_for_prediction(start_logits,
                                                                   end_logits,
                                                                   offset_mapping,
                                                                   max_answer_length,
                                                                   n_best_size,
                                                                   pairwise_decoder,
                                                                   token_is_max_context)

        prelim_predictions.extend(admissible_answers)
    return prelim_predictions, min_null_prediction


def get_admissible_answers_for_prediction(start_logits,
                                          end_logits,
                                          offset_mapping,
                                          max_answer_length,
                                          n_best_size,
                                          pairwise_decoder,
                                          token_is_max_context):
    admissible_answers = []
    # Go through all possibilities for the `n_best_size` greater start and end logits.
    start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
    end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
    #####################
    # we have 2 ways to iterate on this, trying all possible pairs is computationally expensive and can't be implemented as a loss

    if pairwise_decoder:
        # get all possible pairs
        span_iterator = product(start_indexes, end_indexes)
    else:
        # unique start and end positions
        span_iterator = map(sorted, zip(start_indexes, end_indexes))  # to make sure start_index <= end_index

    for start_index, end_index in span_iterator:
        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
        # to part of the input_ids that are not in the context.
        if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or len(offset_mapping[start_index]) < 2
                or offset_mapping[end_index] is None
                or len(offset_mapping[end_index]) < 2
        ):
            continue

        # Don't consider answers with a length that is either < 0 or > max_answer_length.
        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
            continue
        # Don't consider answer that don't have the maximum context available (if such information is
        # provided).
        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
            continue

        admissible_answers.append(
            {
                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                "score": start_logits[start_index] + end_logits[end_index],
                "start_logit": start_logits[start_index],
                "end_logit": end_logits[end_index],
            }
        )

    return admissible_answers


def get_features_map(dataset_features, examples):
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(dataset_features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    return features_per_example
