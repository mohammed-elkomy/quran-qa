from collections import defaultdict
from copy import deepcopy

import numpy as np

from metrics.QQA23_TaskB_eval import normalize_answer_wAr
from text_processing_helpers import get_char_id_to_token_id_map, get_answer_tokens_span, STOP_WORDS, extract_unseen_sequences, remove_prefixes_and_stem


def reject_uninformative_answers(ntop_predictions_pq_id, dataset_jsonl, pq_id):
    """
    uninformative answers:
        • The answer is taken from the question.
        • The whole answer span is a stop word.

    @param ntop_predictions_pq_id: prediction
    @param dataset_jsonl: references
    @param pq_id: id for paragraph query pair
    @return: uninformative answers removed
    """
    pq_dict = [sample for sample in dataset_jsonl if sample["pq_id"] == pq_id][0]
    filtered_entries = []
    ignored_answers = set()
    for entry in ntop_predictions_pq_id:
        answer = normalize_answer_wAr(remove_prefixes_and_stem(entry["answer"])).split()
        question = normalize_answer_wAr(remove_prefixes_and_stem(pq_dict["question"])).split()

        answer_question_inter = set(answer).intersection(set(question))
        if len(answer) < 2:
            if " ".join(answer) in STOP_WORDS:
                ignored_answers.add((entry["answer"], entry["strt_token_indx"], entry["end_token_indx"]))  # The whole answer span is a stop word.
            elif len(answer_question_inter) / (len(set(answer)) + 1e-10) > .9:
                ignored_answers.add((entry["answer"], entry["strt_token_indx"], entry["end_token_indx"]))  # The answer is taken from the question.
            else:
                filtered_entries.append(entry)
        else:
            filtered_entries.append(entry)

    if len(filtered_entries) == 0:  # an empty list after filteration
        return [{"answer": pq_dict["passage"],
                 "score": 1,
                 "rank": 1,
                 "strt_token_indx": 0,
                 "end_token_indx": len(pq_dict["passage"].split()) - 1}]
    return filtered_entries


def post_process_answer_list(ntop_predictions, dataset_jsonl, at_k=20, reject_answers=True):
    ntop_predictions = deepcopy(ntop_predictions)
    for pq_id, answers in ntop_predictions.items():
        samples_with_id = [sample for sample in dataset_jsonl if sample["pq_id"] == pq_id]
        assert samples_with_id  # in case the q_id is not found in the ref dataset, only happens if we are testing the full predictions against a subset of ref dataset
        if samples_with_id and len(answers) != 0:
            sample = samples_with_id[0]
            passage = sample["passage"]
            seen = np.zeros(len(passage.split()), dtype=bool)  # a mask of zeros with length equal to the number of tokens
            char_to_token = get_char_id_to_token_id_map(passage)

            # 1. Handling sub-words +  2. Answer list redundancy elimination are done here
            smart_answers = non_maximum_suppression(answers, char_to_token, passage, seen)

            if reject_answers:
                #   3. Removing uninformative answers:
                #         • The answer is taken from the question.
                #         • The whole answer span is a stop word.
                smart_answers = reject_uninformative_answers(smart_answers, dataset_jsonl, pq_id)

            # sometimes the list might have less than 5 answers, in this case we add any random unseen words
            smart_answers = fill_incomplete_list(passage, seen, smart_answers)

            # 4. Updating the ranked list scores
            update_ranked_list(ntop_predictions, smart_answers, pq_id, at_k)

    return ntop_predictions


def fill_incomplete_list(passage, seen, smart_answers):
    random_guesses = extract_unseen_sequences(seen)
    new_random_guesses = []
    if len(smart_answers) + len(random_guesses) < 10:
        for random_guess in random_guesses:
            if random_guess[0] < random_guess[1] - 1:
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
            "answer": random_guess,
            "score": smart_answers[-1]["score"] - len(smart_answers) * .0001,
            "rank": len(smart_answers) + 1,
            "strt_token_indx": unseen_start,
            "end_token_indx": unseen_end - 1
        })

        tmp_strt_token_indx = smart_answers[-1]["strt_token_indx"]
        tmp_end_token_indx = smart_answers[-1]["end_token_indx"]
        if tmp_end_token_indx < tmp_strt_token_indx:
            raise "xxx"  # TODO REMOVE

    return smart_answers


def update_ranked_list(ntop_predictions, smart_answers, pq_id, at_k, ):
    # we may end up with answers less than the required at_k
    # we just repeat the last answer as this only happens when we have no unseen tokens
    for _ in range(at_k - len(smart_answers)):
        smart_answers.append({
            "answer": smart_answers[-1]["answer"],
            "score": smart_answers[-1]["score"] - len(smart_answers) * .01,
            "rank": len(smart_answers) + 1,
            "strt_token_indx": smart_answers[-1]["strt_token_indx"],
            "end_token_indx": smart_answers[-1]["end_token_indx"],
        })
    for idx, smart_answer in enumerate(smart_answers, start=1):
        smart_answer["rank"] = idx
    ntop_predictions[pq_id] = smart_answers[:at_k]


def non_maximum_suppression(answers, char_to_token, passage, seen):
    # perform the elimination step
    smart_answers = []
    for answer in answers:
        start_token_idx, end_token_idx = get_answer_tokens_span(passage, {"answer": answer["answer"]}, char_to_token)  # start and end token ids
        if start_token_idx is not None:
            answer_seen = seen[start_token_idx:end_token_idx]
            if not all(answer_seen):
                # parts of it are not seen before > find them > add them to the post-processed list of answers
                unseen_sequences = extract_unseen_sequences(answer_seen)
                for unseen_sequence in unseen_sequences:
                    unseen_start, unseen_end = unseen_sequence
                    answer_seen[unseen_start:unseen_end] = True  # mark as seen
                    nearest_answer = passage.split()[start_token_idx + unseen_start:start_token_idx + unseen_end]  # a subset of the answer (the unseen part)
                    nearest_answer = " ".join(nearest_answer)
                    smart_answers.append({
                        "answer": nearest_answer,
                        "score": answer["score"],
                        "rank": len(smart_answers) + 1,
                        "strt_token_indx": start_token_idx + unseen_start,
                        "end_token_indx": start_token_idx + unseen_end - 1
                    })
                    ref_answer = " ".join(passage.split()[smart_answers[-1]["strt_token_indx"]:smart_answers[-1]["end_token_indx"] + 1])
                    assert nearest_answer == ref_answer
    smart_answers = merge_duplicated_answers(smart_answers)
    smart_answers = update_ranks(smart_answers)

    return smart_answers


def merge_duplicated_answers(smart_answers):
    # we may end up with repeated answers after elimination steps
    # we group them together and aggregate the scores for unique occurrences
    unique_smart_answers = {answer["answer"] for answer in smart_answers}
    if len(unique_smart_answers) != len(smart_answers):
        grouped = defaultdict(list)
        for answer in smart_answers:
            grouped[answer["answer"]].append(answer)
        smart_answers = []
        for answer_values in grouped.values():
            total_score = sum(answer_value["score"] for answer_value in answer_values)  # group score for repeated answers
            highest_prop_entry = sorted(answer_values, key=lambda item: item["score"])[-1]
            smart_answers.append({
                "answer": highest_prop_entry["answer"],
                "score": total_score,
                "strt_token_indx": highest_prop_entry["strt_token_indx"],
                "end_token_indx": highest_prop_entry["end_token_indx"],
            })
            tmp_strt_token_indx = smart_answers[-1]["strt_token_indx"]
            tmp_end_token_indx = smart_answers[-1]["end_token_indx"]
            if tmp_end_token_indx < tmp_strt_token_indx:
                raise "bad"  # TODO REMOVE

    return smart_answers


def update_ranks(smart_answers):
    """
    updates the ranks and sorts based on the score
    @param smart_answers:
    @return: updated smart_answers
    """
    smart_answers = sorted(smart_answers, key=lambda item: item["score"], reverse=True)
    for rank, smart_answer in enumerate(smart_answers, start=1):
        smart_answer["rank"] = rank
        tmp_strt_token_indx = smart_answer["strt_token_indx"]
        tmp_end_token_indx = smart_answer["end_token_indx"]
        if tmp_end_token_indx < tmp_strt_token_indx:
            raise "bad"  # TODO REMOVE
    return smart_answers
