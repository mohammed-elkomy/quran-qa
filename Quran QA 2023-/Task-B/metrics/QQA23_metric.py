# Copyright 2020 The HuggingFace Evaluate Authors.
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
""" Quran QA v1.2 metric. """
import json
from pprint import pprint

import datasets

import evaluate
import joblib
import numpy as np

_CITATION = """\
@article{MALHAS2022103068,
title = {Arabic machine reading comprehension on the Holy Qur’an using CL-AraBERT},
journal = {Information Processing & Management},
volume = {59},
number = {6},
pages = {103068},
year = {2022},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2022.103068},
url = {https://www.sciencedirect.com/science/article/pii/S0306457322001704},
author = {Rana Malhas and Tamer Elsayed},
keywords = {Classical Arabic, Reading comprehension, Answer extraction, Partial matching evaluation, Pre-trained language models, Cross-lingual transfer learning},
abstract = {In this work, we tackle the problem of machine reading comprehension (MRC) on the Holy Qur’an to address the lack of Arabic datasets and systems for this important task. We construct QRCD as the first Qur’anic Reading Comprehension Dataset, composed of 1,337 question-passage-answer triplets for 1,093 question-passage pairs, of which 14% are multi-answer questions. We then introduce CLassical-AraBERT (CL-AraBERT for short), a new AraBERT-based pre-trained model, which is further pre-trained on about 1.0B-word Classical Arabic (CA) dataset, to complement the Modern Standard Arabic (MSA) resources used in pre-training the initial model, and make it a better fit for the task. Finally, we leverage cross-lingual transfer learning from MSA to CA, and fine-tune CL-AraBERT as a reader using two MSA-based MRC datasets followed by our QRCD dataset to constitute the first (to the best of our knowledge) MRC system on the Holy Qur’an. To evaluate our system, we introduce Partial Average Precision (pAP) as an adapted version of the traditional rank-based Average Precision measure, which integrates partial matching in the evaluation over multi-answer and single-answer MSA questions. Adopting two experimental evaluation setups (hold-out and cross validation (CV)), we empirically show that the fine-tuned CL-AraBERT reader model significantly outperforms the baseline fine-tuned AraBERT reader model by 6.12 and 3.75 points in pAP scores, in the hold-out and CV setups, respectively. To promote further research on this task and other related tasks on Qur’an and Classical Arabic text, we make both the QRCD dataset and the pre-trained CL-AraBERT model publicly available.}
}

@article{10.1145/3400396,
author = {Malhas, Rana and Elsayed, Tamer},
title = {AyaTEC: Building a Reusable Verse-Based Test Collection for Arabic Question Answering on the Holy Qur’An},
year = {2020},
issue_date = {November 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {6},
issn = {2375-4699},
url = {https://doi.org/10.1145/3400396},
doi = {10.1145/3400396},
abstract = {The absence of publicly available reusable test collections for Arabic question answering on the Holy Qur’an has impeded the possibility of fairly comparing the performance of systems in that domain. In this article, we introduce AyaTEC, a reusable test collection for verse-based question answering on the Holy Qur’an, which serves as a common experimental testbed for this task. AyaTEC includes 207 questions (with their corresponding 1,762 answers) covering 11 topic categories of the Holy Qur’an that target the information needs of both curious and skeptical users. To the best of our effort, the answers to the questions (each represented as a sequence of verses) in AyaTEC were exhaustive—that is, all qur’anic verses that directly answered the questions were exhaustively extracted and annotated. To facilitate the use of AyaTEC in evaluating the systems designed for that task, we propose several evaluation measures to support the different types of questions and the nature of verse-based answers while integrating the concept of partial matching of answers in the evaluation.},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = {oct},
articleno = {78},
numpages = {21},
keywords = {Classical Arabic, evaluation}
}
"""

_DESCRIPTION = """\
QRCD v1.2:
The task is defined as follows: Given a Qur'anic passage that consists of consecutive verses in a specific Surah of the Holy Qur'an, and a free-text question posed in MSA over that passage, a system is required to extract all answers to that question that are stated in the given passage (rather than any answer as in Qur'an QA 2022). Each answer must be a span of text extracted from the given passage. The question can be a factoid or non-factoid question. An example is shown below.
To make the task more realistic (thus challenging), some questions may not have an answer in the given passage. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 10 answer spans.
"""

_KWARGS_DESCRIPTION = """
Computes QRCD v1.2 scores (F1, EM and pAP).
Args:
    predictions: List of triple for question-answers to score with the following elements:
        - the question-answer 'id' field as given in the references (see below)
        - the text of the answer
        - the probability that the question has no answer
    references: List of question-answers dictionaries with the following key-values:
            - 'pq_id': id of the question-answer pair (see above),
            - 'passage': source passage,
            - 'answers': a list of Dict {'text': text of the answer as a string, 'start_char': char index for answer 'text' inside passage}
    no_answer_threshold: float
        Probability threshold to decide that a question has no answer.
Returns:
    'exact': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
    'total': Number of score considered
    'HasAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'HasAns_f1': The F-score of predicted tokens versus the gold answer
    'HasAns_total': Number of score considered
    'NoAns_exact': Exact match (the normalized answer exactly match the gold answer)
    'NoAns_f1': The F-score of predicted tokens versus the gold answer
    'NoAns_total': Number of score considered
    'best_exact': Best exact match (with varying threshold)
    'best_exact_thresh': No-answer probability threshold associated to the best exact match
    'best_f1': Best F1 (with varying threshold)
    'best_f1_thresh': No-answer probability threshold associated to the best F1
Examples:

    
"""


def get_ntop_submission(predictions, cutoff, no_answer_threshold=1000, reject_percent=.05):
    """
    This function applies the threshold given by user no_answer_threshold against the no-answer probabilities returned by the model
     Also truncates answer lists up to cutoff

    :param predictions: raw predictions
    :param cutoff: max answers to be included
    :param no_answer_threshold: threshold for no-answer samples, if higher than 1, no truncation is made (for example the default value 1000)
    if no_answer_threshold is None, the no_answer_threshold is set to mark reject_percent% quantile of the samples as unanswerable
    :param reject_percent: the percent for quantile rejection
    :return:
    """
    if no_answer_threshold is None:
        no_answer_threshold = np.quantile([prediction["no_answer_probability"] for prediction in predictions], 1 - reject_percent)
    ntop_predictions = {}
    for prediction in predictions:
        # collecting top k answers for this pq_id
        answers = [
            {
                "answer": text,
                "score": score,
                "rank": rank,
                "strt_token_indx": start_token_index,
                "end_token_indx": end_token_index,
            }
            for rank, (text, score, start_token_index, end_token_index) in
            enumerate(
                zip(prediction["answers"]["text"],
                    prediction["answers"]["score"],
                    prediction["answers"]["start_token_index"],
                    prediction["answers"]["end_token_index"],
                    ), start=1
            )
        ]
        answers = [a for a in answers if a["strt_token_indx"] is not None]  # remove the no-answer from the list

        for idx, a in enumerate(answers, start=1):
            a["rank"] = idx
        ntop_predictions[prediction["id"]] = answers[:cutoff] \
            if prediction["no_answer_probability"] < no_answer_threshold \
            else []  # truncate and threshold, an empty list is returned when the no_answer_probability is below the threshold we set

    return ntop_predictions


def get_references_format(references):
    dataset_jsonl = []
    for reference in references:
        # extracting reference answers

        answers = [
            {"text": text, "start_char": answer_start}
            for text, answer_start in zip(reference["answers"]["text"], reference["answers"]["answer_start"])
        ]
        dataset_jsonl.append({"pq_id": reference["id"], "answers": answers, "passage": reference["passage"]})
    return dataset_jsonl


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class QQA23(evaluate.Metric):
    def _info(self):

        return evaluate.MetricInfo(
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
                                "rank": datasets.Value("int32"),
                                "score": datasets.Value("float32"),
                                "start_token_index": datasets.Value("int32"),
                                "end_token_index": datasets.Value("int32"),
                            }
                        ),  # A datasets.Sequence with a internal dictionary feature will be automatically converted into a dictionary of lists. This behavior is implemented to have a compatilbity layer with the TensorFlow Datasets library but may be un-wanted in some cases. If you don’t want this behavior, you can use a python list instead of the datasets.Sequence.
                        "no_answer_probability": datasets.Value("float32"),
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
            codebase_urls=["https://gitlab.com/bigirqu/quran-qa-2023"],
            reference_urls=["https://gitlab.com/bigirqu/quran-qa-2023"],
        )

    def _compute(self, predictions, references, cutoff, no_answer_threshold, score_diff=None):
        assert 0 <= no_answer_threshold <= 1
        dataset_jsonl = get_references_format(references)
        ntop_predictions = get_ntop_submission(predictions, cutoff)
        evaluation_result = QQA23_evaluate(dataset_jsonl, ntop_predictions, cutoff)
        if score_diff is None:
            no_answer_probabilities = {p["id"]: p["no_answer_probability"] for p in predictions}
        else:
            no_answer_probabilities = normalize_score_diff(score_diff)

        dataset = [{"paragraphs": [{"qas": references}]}]
        predictions = {p["id"]: p["answers"]["text"][0] if p["answers"]["text"] else "" for p in predictions}

        qid_to_has_ans, qid_to_has_multi_ans = make_qid_to_has_ans(dataset)  # maps qid to True/False
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
        # single and multi answer separation
        has_single_ans_qids = [k for k, v in qid_to_has_multi_ans.items() if not v and k not in no_ans_qids]
        has_multi_ans_qids = [k for k, v in qid_to_has_multi_ans.items() if v]

        exact_raw, f1_raw = get_raw_scores(dataset, predictions)
        pAP_raw = evaluation_result["per_sample"]
        exact_thresh = apply_no_ans_threshold(exact_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        f1_thresh = apply_no_ans_threshold(f1_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        pAP_thresh = apply_no_ans_threshold(pAP_raw, no_answer_probabilities, qid_to_has_ans, no_answer_threshold)
        out_eval = make_eval_dict(exact_thresh, f1_thresh, pAP_thresh)

        if has_ans_qids:
            has_ans_eval = make_eval_dict(exact_thresh, f1_thresh, pAP_thresh, qid_list=has_ans_qids)
            merge_eval(out_eval, has_ans_eval, "HasAns")
        if no_ans_qids:
            no_ans_eval = make_eval_dict(exact_thresh, f1_thresh, pAP_thresh, qid_list=no_ans_qids)
            merge_eval(out_eval, no_ans_eval, "NoAns")
        if has_single_ans_qids:
            single_ans_eval = make_eval_dict(exact_thresh, f1_thresh, pAP_thresh, qid_list=has_single_ans_qids)
            merge_eval(out_eval, single_ans_eval, "SingleAns")
        if has_multi_ans_qids:
            multi_ans_eval = make_eval_dict(exact_thresh, f1_thresh, pAP_thresh, qid_list=has_multi_ans_qids)
            merge_eval(out_eval, multi_ans_eval, "MultiAns")
        find_all_best_thresh(out_eval, predictions, exact_raw, f1_raw, pAP_raw, no_answer_probabilities, qid_to_has_ans)

        for k, v in out_eval.items():
            evaluation_result[k] = v

        assert evaluation_result.get("NoAns_total", 0) + evaluation_result.get("SingleAns_total", 0) + evaluation_result.get("MultiAns_total", 0) == evaluation_result["total"]
        assert evaluation_result.get("HasAns_total", 0) + evaluation_result.get("NoAns_total", 0) == evaluation_result["total"]

        evaluation_result["official_pAP@10"] = float(evaluation_result["official_pAP@10"])
        if score_diff is not None:
            evaluation_result["null_odds_found"] = True
        return evaluation_result


def convert_to_metric_expected_format(dataset_jsonl, ntop_predictions):
    predictions_ = []
    for pq_id, ntop_prediction in ntop_predictions.items():
        predictions_.append({
            "id": pq_id,
            "answers": {
                "text": [answer["answer"] for answer in ntop_prediction],
                "score": [answer["score"] for answer in ntop_prediction],
                "rank": [answer["rank"] for answer in ntop_prediction],
                "start_token_index": [answer["strt_token_indx"] for answer in ntop_prediction],
                "end_token_index": [answer["end_token_indx"] for answer in ntop_prediction],
            },
            "no_answer_probability": 0,
        })
    references_ = []
    for entry in dataset_jsonl:
        references_.append({
            "id": entry["pq_id"],
            "answers": {
                "text": [answer["text"] for answer in entry["answers"]],
                "answer_start": [answer["start_char"] for answer in entry["answers"]],
            },
            "passage": entry["passage"]
        })

    return references_, predictions_


if __name__ == "__main__":
    # to test the metric, the import statements has to be organized like this
    from evaluate import load
    from compute_score_qrcd import (
        apply_no_ans_threshold,
        find_all_best_thresh,
        get_raw_scores,
        make_eval_dict,
        make_qid_to_has_ans,
        merge_eval, normalize_score_diff,
    )

    from metrics.QQA23_TaskB_submission_checker import submission_checker
    from metrics.QQA23_TaskB_eval import load_jsonl, evaluate as QQA23_evaluate

    # simulate the input
    dataset_jsonl = load_jsonl("Original Repo/Task-B/data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl")
    submission_file = "Original Repo/Task-B/data/runs/whole_passage_baseline.json"

    #
    # predictions_ = [
    #     {
    #         'id': '13:18-24_360',
    #         'answers': {
    #             "text": ["للذين استجابوا لربهم الحسنى والذين لم يستجيبوا له لو أن لهم ما في الأرض جميعا ومثله معه لافتدوا به أولئك لهم سوء الحساب ومأواهم جهنم وبئس المهاد . أفمن يعلم أنما أنزل إليك من ربك الحق كمن هو أعمى إنما يتذكر أولو الألباب . الذين يوفون بعهد الله ولا ينقضون الميثاق . والذين يصلون ما أمر الله به أن يوصل ويخشون ربهم ويخافون سوء الحساب . والذين صبروا ابتغاء وجه ربهم وأقاموا الصلاة وأنفقوا مما رزقناهم سرا وعلانية ويدرءون بالحسنة السيئة أولئك لهم عقبى الدار . جنات عدن يدخلونها ومن صلح من آبائهم وأزواجهم وذرياتهم والملائكة يدخلون عليهم من كل باب . سلام عليكم بما صبرتم فنعم عقبى الدار ."],
    #             "rank": [0],
    #             "score": [100],
    #             "start_token_index": [0],
    #             "end_token_index": [109],
    #         },
    #         'no_answer_probability': 0
    #     },
    #     {
    #         'id': '28:85-88_322',
    #         'answers': {
    #             "text": [],
    #             "rank": [],
    #             "score": [],
    #             "start_token_index": [],
    #             "end_token_index": [],
    #         },
    #         'no_answer_probability': 1.0
    #     }
    # ]
    # references_ = [
    #     {
    #         "id": "13:18-24_360",
    #         "passage": "للذين استجابوا لربهم الحسنى والذين لم يستجيبوا له لو أن لهم ما في الأرض جميعا ومثله معه لافتدوا به أولئك لهم سوء الحساب ومأواهم جهنم وبئس المهاد . أفمن يعلم أنما أنزل إليك من ربك الحق كمن هو أعمى إنما يتذكر أولو الألباب . الذين يوفون بعهد الله ولا ينقضون الميثاق . والذين يصلون ما أمر الله به أن يوصل ويخشون ربهم ويخافون سوء الحساب . والذين صبروا ابتغاء وجه ربهم وأقاموا الصلاة وأنفقوا مما رزقناهم سرا وعلانية ويدرءون بالحسنة السيئة أولئك لهم عقبى الدار . جنات عدن يدخلونها ومن صلح من آبائهم وأزواجهم وذرياتهم والملائكة يدخلون عليهم من كل باب . سلام عليكم بما صبرتم فنعم عقبى الدار .",
    #         "answers":
    #             [
    #                 {
    #                     "text": "جنات عدن يدخلونها ومن صلح من آبائهم وأزواجهم وذرياتهم",
    #                     "answer_start": 456
    #                 }
    #             ],
    #     },
    #     {"id": "28:85-88_322",
    #      "passage": "إن الذي فرض عليك القرآن لرادك إلى معاد قل ربي أعلم من جاء بالهدى ومن هو في ضلال مبين . وما كنت ترجو أن يلقى إليك الكتاب إلا رحمة من ربك فلا تكونن ظهيرا للكافرين . ولا يصدنك عن آيات الله بعد إذ أنزلت إليك وادع إلى ربك ولا تكونن من المشركين . ولا تدع مع الله إلها آخر لا إله إلا هو كل شيء هالك إلا وجهه له الحكم وإليه ترجعون .",
    #      "answers": []}
    # ]

    submission_checker(submission_file)
    with open(submission_file) as ntop_prediction_file:
        ntop_predictions = json.load(ntop_prediction_file)

    references_, predictions_ = convert_to_metric_expected_format(dataset_jsonl, ntop_predictions)
    # testing yourself
    # qrcd_metric = load(__file__)  # testing yourself
    # results = qrcd_metric.compute(predictions=predictions_, references=references_, cutoff=10, no_answer_threshold=0)

    results = QQA23()._compute(predictions=predictions_, references=references_, cutoff=10, no_answer_threshold=0)  # for debugging break points

    results.pop("per_sample")
    pprint(results)
    exit()

from .compute_score_qrcd import (
    apply_no_ans_threshold,
    find_all_best_thresh,
    get_raw_scores,
    make_eval_dict,
    make_qid_to_has_ans,
    merge_eval, normalize_score_diff
)

from .QQA23_TaskB_eval import evaluate as QQA23_evaluate
