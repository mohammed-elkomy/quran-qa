import json
import os
import shutil
import traceback

import joblib
from transformers import (
    EvalPrediction,
)

from answer_decoding.bertlike_decoder import decode_model_logits
from metrics.QQA23_metric import get_ntop_submission


def post_processing_function(examples, features,
                             logits_predictions,
                             data_args, training_args,
                             log_level,
                             answer_column_name, context_column_name,
                             stage="eval"):
    ######
    # stage has three cases
    # "train-eval" : train dataset evaluated at the end of each epoch to detect overfitting just by looking at tensorboard curves
    # "eval" : evaluate dataset
    # "test" : test dataset
    ######
    # Post-processing: we match the start logits and end logits to answers in the original context.
    top_k_predictions, scores_diff_json = decode_model_logits(
        unprocessed_dataset=examples,
        processed_dataset=features,
        predictions=logits_predictions,
        n_best_size=data_args.n_best_size,
        pairwise_decoder=training_args.pairwise_decoder,
        max_answer_length=data_args.max_answer_length,
        null_score_diff_threshold=data_args.null_score_diff_threshold,
        log_level=log_level,
    )

    formatted_predictions = format_predictions_QQA23(top_k_predictions, cut_off=data_args.metric_cutoff)  # doesn't truncate answers list

    model_dump_path = make_dump_file(examples, features, stage, formatted_predictions, scores_diff_json, training_args)

    submission_format = get_ntop_submission(formatted_predictions, data_args.metric_cutoff, no_answer_threshold=data_args.no_answer_threshold)  # truncates answers list

    with open(model_dump_path.replace(".dump", "-submission.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(submission_format))

    has_no_answers = all(
        len(answer["text"]) == 0 for answer in examples[answer_column_name]
    )

    if not has_no_answers:
        # if we have labelled data we can compute metrics
        references = [{"id": ex["id"], "answers": ex[answer_column_name], "passage": ex[context_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    else:
        print(f"For {stage} stage: no evaluation made (blind data)")

def format_predictions_QQA23(top_k_predictions, cut_off):
    # Format the result to the format expected by metric evaluation script.
    formatted_predictions = []
    # QRCD v2 metric format
    for pred_id, top_k_prediction in top_k_predictions.items():
        answers_predictions = {
            "text": [k_th_prediction["text"] for k_th_prediction in top_k_prediction],
            "score": [k_th_prediction["probability"] for k_th_prediction in top_k_prediction],
            "rank": list(range(1, len(top_k_prediction) + 1)),
            "start_token_index": [k_th_prediction["start_token_index"] for k_th_prediction in top_k_prediction],
            "end_token_index": [k_th_prediction["end_token_index"] for k_th_prediction in top_k_prediction],
        }

        probability = [1 - (min(cut_off, idx) / cut_off)
                       for idx, text in
                       enumerate(answers_predictions["text"]) if text == ""]
        if len(probability) > 1 and len(probability) != len(top_k_prediction):
            joblib.dump({
                "top_k_prediction": top_k_prediction,
            }, f"more than zero.dmp", compress=6)

        formatted_predictions.append({"id": pred_id,
                                      "answers": answers_predictions,
                                      "no_answer_probability": max(probability)})

    return formatted_predictions


def make_dump_file(examples, features, stage, formatted_predictions, scores_diff_json, training_args):
    model_dump_path = os.path.join(training_args.my_output_dir, f"{stage}-{training_args.seed}.dump")
    if stage != "train-eval":  # we don't make a dump file for training data
        try:
            joblib.dump({
                "formatted_predictions": formatted_predictions,
                "scores_diff_json": scores_diff_json,
                # "examples": list(examples),
                # "features": list(features),
            }, f"{model_dump_path}", compress=6)
            print(f"{'=' * 50}\ndump file saved to {model_dump_path}\n{'=' * 50}")
        except:
            print(traceback.format_exc())
    return model_dump_path
