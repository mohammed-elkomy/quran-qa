import json
import re
import re
import string
import sys
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer
from matplotlib import pyplot as plt

from data.qrcd_eval import find_all_occurences, pRR_max_over_ground_truths


def per_sample_evaluate(dataset_jsonl, ntop_predictions):
    lengths = {len(value) for value in ntop_predictions.values()}
    assert len(lengths) == 1  # all ntop_predictions return exactly k predictions
    k = list(lengths)[0]

    total = 0
    pRR_scores = []
    for pq_dict in dataset_jsonl:
        total += 1
        pq_id = pq_dict['pq_id']

        if pq_id not in ntop_predictions:
            message = 'Unanswered question ' + pq_id + \
                      ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = pq_dict['answers']
        passage = pq_dict['passage']

        ntop_predictions_4pq = ntop_predictions[pq_id]
        predictions = []
        all_predictions_start_char_list = []  # contains the start_char occurences for each predicted answer
        for i in range(len(ntop_predictions_4pq)):
            answer = ntop_predictions_4pq[i]['answer']
            predictions.append(answer)
            start_char_occurences = find_all_occurences(passage, answer)
            all_predictions_start_char_list.append(start_char_occurences)

        pRR = pRR_max_over_ground_truths(predictions, all_predictions_start_char_list, ground_truths)
        pRR_scores.append(pRR)

    return pRR_scores


def get_char_id_to_token_id_map(text):
    token_id = 0
    token_id_map = []
    for char in text:
        if char == " ":
            token_id += 1
        token_id_map.append(token_id)

    return token_id_map


farasa_segmenter = FarasaSegmenter(interactive=True)
stemmer = FarasaStemmer(interactive=True)


def normalize_text(s):
    """remove punctuation, some stopwords and extra whitespace."""

    def remove_stopWords(text):
        terms = []
        stopWords = {'من', 'الى', 'إلى', 'عن', 'على', 'في', 'حتى'}
        for term in text.split():
            if term not in stopWords:
                terms.append(term)
        return " ".join(terms)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # Arabic punctuation
        exclude.add('،')
        exclude.add('؛')
        exclude.add('؟')
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_stopWords(remove_punc(s)))


def remove_prefixes_and_stem(text):
    text_tokens = farasa_segmenter.segment(text).split()
    tokens = []
    for token in text_tokens:
        token = re.sub(r'^و[+]', '', token)  # only begining of words
        token = re.sub(r'^ف[+]', '', token)
        token = re.sub(r'^ب[+]', '', token)
        token = re.sub(r'^ك[+]', '', token)
        token = re.sub(r'^ل[+]', '', token)
        token = re.sub(r'^لل[+]', '', token)
        token = re.sub(r'^ال[+]', '', token)

        # defragment by removing pluses
        token = re.sub(r'[+]', '', token)
        tokens.append(token)

    return stemmer.stem(" ".join(tokens))


def reject_uninformative_answers(ntop_predictions_pq_id, dataset_jsonl, pq_id):
    pq_dict = [sample for sample in dataset_jsonl if sample["pq_id"] == pq_id][0]
    filtered_entries = []
    ignored_answers = set()
    for entry in ntop_predictions_pq_id:
        answer = normalize_text(remove_prefixes_and_stem(entry["answer"])).split()
        question = normalize_text(remove_prefixes_and_stem(pq_dict["question"])).split()

        answer_question_inter = set(answer).intersection(set(question))
        if len(answer) < 2:
            if " ".join(answer) in {"إذ", "", "بعد", "ها", "قال", "هذا", "ه", "اذا", "يا", "إن", "لكن", "ليس",
                                    "أما", "الله", "أو", "ما", "لا", "قد", "ثم", "هم", "أن", "ذلك", "أي", "يومئذ",
                                    "مص", "هو", "مر", "كم", "كان", "قبل", "فوق", "بل", "الذي", "إلا", "إذا", "أفمن", "بين"
                                    }:
                ignored_answers.add((entry["answer"], pq_dict["question"]))
                # removed.add(entry["answer"])
            # removed.add(" ".join(answer))
            # else
            #     kept.add((" ".join(answer), entry["answer"]))
            # kept.add(entry["answer"])
            elif len(answer_question_inter) / (len(set(answer)) + 1e-10) > .9:
                ignored_answers.add((entry["answer"], pq_dict["question"]))
            else:
                filtered_entries.append(entry)
        else:
            filtered_entries.append(entry)

    for ignored_answer, _ in ignored_answers:
        if ignored_answer not in " ".join(filtered_entry["answer"] for filtered_entry in filtered_entries):
            for filtered_entry in filtered_entries:
                if filtered_entry["answer"] + " " + ignored_answer in pq_dict["passage"]:
                    filtered_entry["answer"] = filtered_entry["answer"] + " " + ignored_answer
                    break
                elif ignored_answer + " " + filtered_entry["answer"] in pq_dict["passage"]:
                    filtered_entry["answer"] = ignored_answer + " " + filtered_entry["answer"]
                    break
    return filtered_entries


def get_answer_tokens_span(passage, answer, char_to_token):
    answer_start_char = passage.index(answer["answer"])
    answer_end_char = answer_start_char + len(answer["answer"])
    span_token_intersection = Counter(char_to_token[answer_start_char:answer_end_char])
    answer_tokens = []
    for token_id, token_length in Counter(char_to_token).items():
        token_intersection_length = span_token_intersection.get(token_id, 0)
        if token_intersection_length > .5 * token_length:  # half of the token was selected, I will include it in the heatmap
            answer_tokens.append(token_id)
    # passage.split()[min(answer_tokens):max(answer_tokens) + 1]
    if not answer_tokens:
        return None, None

    # adding one to the last token index, since slice operator is exclusive
    return min(answer_tokens), max(answer_tokens) + 1


def smart_pRR_pruning(ntop_predictions, dataset_jsonl, at_k=20, reject_answers=False):
    ntop_predictions = deepcopy(ntop_predictions)
    for pq_id, answers in ntop_predictions.items():
        sample = [sample for sample in dataset_jsonl if sample["pq_id"] == pq_id][0]
        passage = sample["passage"]
        seen = np.zeros(len(passage.split()), dtype=bool)
        char_to_token = get_char_id_to_token_id_map(passage)

        smart_answers = []

        for answer in answers:
            start_token_idx, end_token_idx = get_answer_tokens_span(passage, answer, char_to_token)
            if start_token_idx is not None:
                answer_seen = seen[start_token_idx:end_token_idx]
                if not all(answer_seen):
                    unseen_sequences = extract_unseen_sequences(answer_seen)
                    for unseen_sequence in unseen_sequences:
                        unseen_start, unseen_end = unseen_sequence
                        answer_seen[unseen_start:unseen_end] = True
                        nearest_answer = passage.split()[start_token_idx + unseen_start:start_token_idx + unseen_end]
                        nearest_answer = " ".join(nearest_answer)
                        smart_answers.append({
                            "answer": nearest_answer, "score": answer["score"], "rank": len(smart_answers) + 1
                        })
        if len({answer["answer"] for answer in smart_answers}) != len(smart_answers):
            grouped = defaultdict(list)
            for answer in smart_answers:
                grouped[answer["answer"]].append(answer)
            smart_answers = []
            for answer_values in grouped.values():
                total_score = sum(answer_value["score"] for answer_value in answer_values)
                answer = answer_values[0]["answer"]
                smart_answers.append({
                    "answer": answer, "score": total_score,
                })

            smart_answers = sorted(smart_answers, key=lambda item: item["score"], reverse=True)
            for rank, smart_answer in enumerate(smart_answers, start=1):
                smart_answer["rank"] = rank

        if reject_answers:
            smart_answers = reject_uninformative_answers(smart_answers, dataset_jsonl, pq_id)

            random_guesses = extract_unseen_sequences(seen)
            new_random_guesses = []
            if len(smart_answers) + len(random_guesses) < 5:
                for random_guess in random_guesses:
                    mid = (random_guess[0] + random_guess[1]) // 2
                    new_random_guesses.append((random_guess[0], mid))
                    new_random_guesses.append((mid, random_guess[1]))
            if new_random_guesses:
                random_guesses = new_random_guesses

            for random_guess in random_guesses:
                unseen_start, unseen_end = random_guess
                random_guess = passage.split()[unseen_start:unseen_end]
                random_guess = " ".join(random_guess)
                smart_answers.append({
                    "answer": random_guess, "score": smart_answers[-1]["score"] - len(smart_answers) * .0001, "rank": len(smart_answers) + 1
                })
        if pq_id == "24:46-54_415":
            print()
        for _ in range(at_k - len(smart_answers)):
            smart_answers.append({
                "answer": smart_answers[-1]["answer"], "score": smart_answers[-1]["score"] - len(smart_answers) * .0001, "rank": len(smart_answers) + 1
            })

        for idx, smart_answer in enumerate(smart_answers, start=1):
            smart_answer["rank"] = idx
        ntop_predictions[pq_id] = smart_answers[:at_k]

    return ntop_predictions


def extract_unseen_sequences(subset):
    start = 0
    sequences = []
    for idx, value in enumerate(subset[1:], start=1):
        if value != subset[start]:
            if not any(subset[start:idx]):
                sequences.append((start, idx))
            start = idx
    if not any(subset[start:]):
        sequences.append((start, len(subset)))
    return sequences


def draw_frequency(optimal_indexes_frequency_k, k_cutoff, path):
    optimal_indexes_frequency_k = Counter(optimal_indexes_frequency_k.values())
    optimal_indexes_frequency_k = sorted(optimal_indexes_frequency_k.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 1000)  # sort by index 0,1,2,3.. as i converted them to strings
    df = pandas.DataFrame.from_dict({
        int(key) + 1 if key.isdigit() else key  # one based instead of zero based
        : value
        for key, value in optimal_indexes_frequency_k}, orient='index')
    df.plot(figsize=(9, 7), kind='bar', rot=30, fontsize=13, title=f"The histogram for the rank at which the model achieves the highest pRR@{k_cutoff} per sample", legend=False)
    plt.savefig(path, dpi=400)
