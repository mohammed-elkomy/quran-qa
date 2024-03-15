from collections import Counter

from farasa.stemmer import FarasaStemmer

from metrics.QQA23_TaskB_eval import remove_prefixes_in_text

stemmer = FarasaStemmer(interactive=True)

STOP_WORDS = {"إذ", "", "بعد", "ها", "قال", "هذا", "ه", "اذا", "يا", "إن", "لكن", "ليس",
              "أما", "أو", "ما", "لا", "قد", "ثم", "هم", "أن", "ذلك", "أي", "يومئذ",
              "مص", "هو", "مر", "كم", "كان", "قبل", "فوق", "بل", "الذي", "إلا", "إذا", "أفمن", "بين",
              "الم", "."
              }


def remove_prefixes_and_stem(text):
    removed_prefixes = remove_prefixes_in_text(text)  # clean arabic text
    return stemmer.stem(removed_prefixes)  # stem arabic text for relaxed matching


def get_char_id_to_token_id_map(text):
    """
    @param text: a passage string
    @return: a list mapping each character in @text to its token index
    """
    token_id = 0
    token_id_map = []
    for char in text:
        if char == " ":
            token_id += 1
        token_id_map.append(token_id)

    return token_id_map


def get_answer_tokens_span(passage, answer, char_to_token):
    """
        1. Handling sub-words is done here
    @param passage: the passage text
    @param answer: the answer text
    @param char_to_token: char to token index map
    @return: start token index to end token index for the answer text
    """
    try:
        answer_start_char = passage.index(answer["answer"])
    except:
        print("issue", answer["answer"])
        return None, None

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


def extract_unseen_sequences(subset):
    """
    finds continuous intervals of zeros
    @param subset: a subset of a boolean mask
    @return: zero sequences
    """
    start = 0
    sequences = []
    for idx, value in enumerate(subset[1:], start=1):
        if value != subset[start]:
            if not any(subset[start:idx]):  # subset[start:idx] is only zeros => unseen
                sequences.append((start, idx))
            start = idx  # next interval

    # last interval
    if not any(subset[start:]):
        sequences.append((start, len(subset)))
    return sequences


def verify_positional_correctness(ntop_predictions, dataset_jsonl):
    for sample_preds in ntop_predictions:
        pq_id = sample_preds["id"]
        passage = {sample_ref["passage"] for sample_ref in dataset_jsonl if sample_ref["pq_id"] == pq_id}
        assert len(passage) == 1
        passage = list(passage)[0]
        for token_start_idx, token_end_idx, answer_text in zip(sample_preds["answers"]["start_token_index"],
                                                               sample_preds["answers"]["end_token_index"],
                                                               sample_preds["answers"]["text"],
                                                               ):
            if token_start_idx is None:
                assert answer_text == ""
            else:
                ref_answer = passage.split()[token_start_idx:token_end_idx + 1]
                assert answer_text == " ".join(ref_answer)
