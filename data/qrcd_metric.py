# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SQuAD metric. """
import json
import os.path
import tempfile
from collections import defaultdict
from pprint import pprint

import datasets

_CITATION = """\
QRCD
"""

_DESCRIPTION = """
QRCD
"""

_KWARGS_DESCRIPTION = """
QRCD
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Squad(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "score": datasets.Value("float32"),
                            }
                        ),
                    },
                    "references": {
                        "id": datasets.Value("string"),
                        "passage": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            codebase_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
            reference_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        )

    def _compute(self, predictions, references, model_name_or_path):
        dataset_jsonl = []
        for reference in references:
            # extracting reference answers

            answers = [
                {"text": text, "start_char": answer_start}
                for text, answer_start in zip(reference["answers"]["text"], reference["answers"]["answer_start"])
            ]
            dataset_jsonl.append({"pq_id": reference["id"], "answers": answers, "passage": reference["passage"]})

        ntop_predictions = {}
        for prediction in predictions:
            # collecting top k answers for this pq_id
            answers = [
                {"answer": text, "score": score, "rank": rank}
                for rank, (text, score) in enumerate(
                    zip(prediction["answers"]["text"], prediction["answers"]["score"]), start=1
                )
            ]

            ntop_predictions[prediction["id"]] = answers[:5]  # 5 answers maximum for pRR

        if os.path.exists(model_name_or_path):
            ntop_predictions_json = os.path.join(model_name_or_path, "ntop_predictions.json")
        else:
            ntop_predictions_json = f"{os.path.split(model_name_or_path)[-1]}-ntop_predictions.json"

        with open(ntop_predictions_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(ntop_predictions))

        if format_checker(ntop_predictions_json) is False:
            print("Please review the above warning(s) or error message(s) related to this run file.")
        else:
            return evaluate(dataset_jsonl=dataset_jsonl, ntop_predictions=ntop_predictions)
        return {'pRR': "bad submission", 'exact_match': "bad submission", 'f1': "bad submission"}


if __name__ == "__main__":
    # to test the metric, the import statements has to be organized like this
    from datasets import load_metric
    from qrcd_eval import evaluate, load_jsonl
    from qrcd_format_checker import format_checker

    # simulate the input
    dataset_jsonl = load_jsonl("qrcd/qrcd_v1.1_dev.jsonl")
    format_checker("qrcd/sample_submission.json")
    with open("qrcd/sample_submission.json") as ntop_prediction_file:
        ntop_predictions = json.load(ntop_prediction_file)
    formatted_predictions = []
    for pq_id, ntop_prediction in ntop_predictions.items():
        formatted_predictions.append({
            "id": pq_id,
            "answers": {
                "text": [answer["answer"] for answer in ntop_prediction],
                "score": [answer["score"] for answer in ntop_prediction],
            }
        })
    references_ = []
    for entry in dataset_jsonl:
        references_.append({
            "id": entry["pq_id"],
            "answers": {
                "text": [answer["text"] for answer in entry["answers"]],
                "answer_start": [answer["start_char"] for answer in entry["answers"]],
            }
        })
    metric = load_metric(__file__)  # testing yourself
    print(metric.compute(predictions=formatted_predictions, references=references_))
    exit()  # no need to do those imports below

# this import statement has to be kept here
from .qrcd_eval import evaluate
from .qrcd_format_checker import functional_checker as format_checker
