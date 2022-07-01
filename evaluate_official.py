"""
this file will run the official evaluation script on the submission files
post_processing/results/eval/ensemble_keep.json
post_processing/results/eval/ensemble_original.json
post_processing/results/eval/ensemble_remove.json

post_processing/results/test/ensemble_keep.json
post_processing/results/test/ensemble_original.json
post_processing/results/test/ensemble_remove.json
"""
import json

from data.qrcd_eval import load_jsonl, evaluate

##################################
# eval data
##################################
print("EVAL DATA")
eval_dataset_jsonl = load_jsonl("data/qrcd/qrcd_v1.1_dev.jsonl")

with open("post_processing/results/eval/ensemble_original.json", 'r', encoding='utf-8') as ensemble_original_submission_file:
    ensemble_original_submission_file = json.load(ensemble_original_submission_file)

original_eval_results = evaluate(eval_dataset_jsonl, ensemble_original_submission_file)

with open("post_processing/results/eval/ensemble_keep.json", 'r', encoding='utf-8') as ntop_predictions_keep:
    ntop_predictions_keep = json.load(ntop_predictions_keep)

keep_uninformative_eval_results = evaluate(eval_dataset_jsonl, ntop_predictions_keep)

with open("post_processing/results/eval/ensemble_remove.json", 'r', encoding='utf-8') as ntop_predictions_reject:
    ntop_predictions_reject = json.load(ntop_predictions_reject)

reject_uninformative_eval_results = evaluate(eval_dataset_jsonl, ntop_predictions_reject)

print("original results, without post-processing")
print(original_eval_results)

print("post-processing with uninformative answers kept")
print(keep_uninformative_eval_results)

print("post-processing with uninformative answers removed")
print(reject_uninformative_eval_results)
print("=" * 50)
##################################
# test data
# the organizers decided to drop 40 samples from the official test data split for fair comparison among teams
# for that reason my submission files will contain 280 samples while the official gold answers file only has 240
##################################
print("OFFICIAL TEST DATA")
test_dataset_jsonl = load_jsonl("data/qrcd/qrcd_v1.1_test_gold.jsonl")
with open("post_processing/results/test/ensemble_original.json", 'r', encoding='utf-8') as ensemble_original_submission_file:
    ensemble_original_submission_file = json.load(ensemble_original_submission_file)

original_eval_results = evaluate(test_dataset_jsonl, ensemble_original_submission_file)

with open("post_processing/results/test/ensemble_keep.json", 'r', encoding='utf-8') as ntop_predictions_keep:
    ntop_predictions_keep = json.load(ntop_predictions_keep)

keep_uninformative_eval_results = evaluate(test_dataset_jsonl, ntop_predictions_keep)

with open("post_processing/results/test/ensemble_remove.json", 'r', encoding='utf-8') as ntop_predictions_reject:
    ntop_predictions_reject = json.load(ntop_predictions_reject)

reject_uninformative_eval_results = evaluate(test_dataset_jsonl, ntop_predictions_reject)

print("original results, without post-processing")
print(original_eval_results)

print("post-processing with uninformative answers kept")
print(keep_uninformative_eval_results)

print("post-processing with uninformative answers removed")
print(reject_uninformative_eval_results)
print("=" * 50)
