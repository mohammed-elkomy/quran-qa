import os
import random

import joblib

MAX_ANSWER_SCORE = 1.0

PAD_TOKEN = 0

MAX_TRIES = 250


def get_start_and_end_of_answer(start_char, end_char, token_start_index, token_end_index, offsets):
    # move the token_start_index and token_end_index to the two ends of the answer.
    # Note: we could go after the last offset if the answer is the last word (edge case).
    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
        token_start_index += 1

    while offsets[token_end_index][1] >= end_char and token_end_index >= 0:
        token_end_index -= 1

    # token_start_index can be -1 and token_end_index is +1 at the same time, this means the first token is the answer
    return token_start_index, token_end_index


def get_start_and_end_of_passage(input_ids, pad_on_right, sequence_ids):
    # since the question and passage are packed together as "QQQQQQQQQPPPPPPPPPP" for pad_on_right = True and sequence_ids are 0000000011111111111
    # Start token index of the current span in the text.
    token_start_index = 0
    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
        token_start_index += 1
    # End token index of the current span in the text.
    token_end_index = len(input_ids) - 1
    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
        token_end_index -= 1
    return token_start_index, token_end_index


def log_tokenization_mistakes(context, answers, start_char, end_char, token_start_index, token_end_index, offsets, data_logs_file):
    # after tokenization, we may end up with an answer was broken words (the answer span can't be exactly represented by the tokenization result)
    if offsets[token_end_index + 1][1] != end_char or offsets[token_start_index - 1][0] != start_char:
        print("=" * 50, file=data_logs_file)
        print("the tokenizer can't set the boundaries", file=data_logs_file)
        tokenizer_start_char = offsets[token_start_index - 1][0]
        tokenizer_end_char = offsets[token_end_index + 1][1]

        print(f"tokenizer:{context[tokenizer_start_char:tokenizer_end_char]}", file=data_logs_file)
        print(f"label slice:{context[start_char:end_char]}", file=data_logs_file)

        print(f"label:{answers}", file=data_logs_file)
        print("=" * 50, file=data_logs_file, flush=True)


def common_preprocessing(context_column_name, data_args, examples, max_seq_length, pad_on_right, question_column_name, tokenizer):
    """
    This function does the common tokenization business
    @param context_column_name:
    @param data_args:
    @param examples:
    @param max_seq_length:
    @param pad_on_right:
    @param question_column_name:
    @param tokenizer: a FastTokenizer like bert fast tokenizer
    @return:
    """

    # Some questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]  # this has no effect on the answer span
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.

    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",  # truncate only the context, the question and passage are packed together as "QQQQQQQQQPPPPPPPPPP" for pad_on_right = True
        max_length=max_seq_length,  # for the whole pair
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    return tokenized_examples, sample_mapping


# Training preprocessing: the features returned by this function is used for loss computation and training
# not evaluating metrics (check preprocess_validation_features)
def tokenize_train_features(examples,
                            question_column_name,
                            context_column_name,
                            tokenizer,
                            max_seq_length,
                            pad_on_right,
                            data_args,
                            ):
    num_samples = len(examples["id"])
    # this preprocessing is shared between validation features and train features
    # (it does the tokenization process and question cleaning)
    # this step doesn't add labels for training, labels are added in different step to allow us to update labels with every epoch
    tokenized_examples, sample_mapping = common_preprocessing(context_column_name,
                                                              data_args,
                                                              examples,
                                                              max_seq_length,
                                                              pad_on_right,
                                                              question_column_name,
                                                              tokenizer)

    tokenized_examples["sequence_ids"] = [encoding.sequence_ids for encoding in tokenized_examples.encodings]

    examples_with_tokenization_features = examples
    feature_names = tokenized_examples.keys()  # a generic list of feature_names

    for feature_name in feature_names:
        examples_with_tokenization_features[feature_name] = [[] for _ in range(num_samples)]  # placeholder list

    for feature_idx, sample_idx in enumerate(sample_mapping):
        # One example can give several spans, this is the index of the example containing this span of text.
        # feature_idx != sample_idx, won't be equal for long sequences
        for feature_name in feature_names:
            # grouping features for every sample
            feature_value_at_idx = tokenized_examples[feature_name][feature_idx]
            examples_with_tokenization_features[feature_name][sample_idx].append(feature_value_at_idx)

    assert all(
        len(v) == num_samples
        for v in examples_with_tokenization_features.values()
    ), "Not All values have the same length"

    return examples_with_tokenization_features


def generate_training_labels(examples_with_tokenization_features,
                             tokenizer,
                             answer_column_name,
                             context_column_name,
                             data_args,
                             pad_on_right,
                             data_logs_file, ):
    feature_keys = examples_with_tokenization_features.keys()
    feature_values = examples_with_tokenization_features.values()

    start_positions, end_positions = [], []  # labels for all samples, each sample will be a list
    # an answer score of 1 means the full score

    for sample_index, sample_features in enumerate(zip(*feature_values)):
        sample_features = dict(zip(feature_keys, sample_features))

        context = examples_with_tokenization_features[context_column_name][sample_index]

        sample_start_positions, sample_end_positions = get_labels_for_sample(
            tokenizer,
            context, sample_features,
            answer_column_name, pad_on_right, data_args, data_logs_file,
        )  # this will get start_positions and end_positions labels for a dataset example,
        # each one of them is a list of labels in case the text is longer than the max_text_length, tokenization will break long sequences into parts
        # and each part will have it's corresponding label whether the answer exists or not in the broken context

        start_positions.append(sample_start_positions)
        end_positions.append(sample_end_positions)


    examples_with_tokenization_features["start_positions"] = start_positions
    examples_with_tokenization_features["end_positions"] = end_positions

    feature_columns = [k for k, v in examples_with_tokenization_features.items() if type(v[0]) == list]  # column names with processed features, input features are lists of strings rather than list of lists(tokenized)
    return {
        feature_name: sum(examples_with_tokenization_features[feature_name], []) for
        # flatten features, each feature now represents a full context or portions of longer contexts with their corresponding labels (if the answer outside the context portion cls_index is used as a label, which means it's unanswerable)
        feature_name in feature_columns
    }


def get_labels_for_sample(tokenizer,
                          context, sample_features,
                          answer_column_name, pad_on_right, data_args, data_logs_file,
                          ):
    num_text_splits = len(sample_features["offset_mapping"])

    # Let's label those examples!
    sample_start_positions = []
    sample_end_positions = []
    sample_answers_scores = []
    for text_split_idx in range(num_text_splits):
        input_ids = sample_features["input_ids"][text_split_idx]
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = sample_features["sequence_ids"][text_split_idx]

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offsets = sample_features["offset_mapping"][text_split_idx]

        # We will label impossible answers with the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)

        answers = sample_features[answer_column_name]
        if len(answers["answer_start"]) == 0:
            # If no answers are given, set the cls_index as answer.
            sample_start_positions.append([cls_index])
            sample_end_positions.append([cls_index])
        else:
            answers_start_positions, answers_end_positions = get_multi_answer_labels(input_ids, sequence_ids, offsets, answers, cls_index, context, data_args, data_logs_file, pad_on_right, )
            sample_start_positions.append(answers_start_positions)
            sample_end_positions.append(answers_end_positions)

    return sample_start_positions, sample_end_positions


def get_multi_answer_labels(input_ids, sequence_ids, offsets, answers, cls_index, context, data_args, data_logs_file, pad_on_right, ):
    # here we return a list of labels for every entry, this means we support questions with multi-answer
    answers_start_positions = []
    answers_end_positions = []

    for start_char, answer_text in zip(answers["answer_start"], answers["text"]):  # multi answer support implemented here
        # Start/end character index of the answer in the text.
        end_char = start_char + len(answer_text)
        token_start_index, token_end_index = get_start_and_end_of_passage(input_ids, pad_on_right, sequence_ids)

        # Detect if the answer is not found in this doc split, due to truncation (in which case this feature is labeled with the CLS index).
        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            # print("OUT OF SPAN")
            answers_start_positions.append(cls_index)
            answers_end_positions.append(cls_index)
        else:
            assert offsets[token_start_index][0] <= start_char <= end_char <= offsets[token_end_index][1], "an answer is supposed to be in the span"

            token_start_index, token_end_index = get_start_and_end_of_answer(start_char, end_char, token_start_index, token_end_index, offsets)

            log_tokenization_mistakes(context, answers, start_char, end_char, token_start_index, token_end_index, offsets, data_logs_file)

            answers_start_positions.append(token_start_index - 1)  # because of the while loop stopping condition at get_start_and_end_of_answer
            answers_end_positions.append(token_end_index + 1)  # because of the while loop stopping condition at get_start_and_end_of_answer

    if not answers_start_positions:
        # empty answers list, will never happen, just a sanity check
        joblib.dump((input_ids, sequence_ids, offsets, answers, cls_index, context, data_args, data_logs_file, pad_on_right), "empty answers list.dmp", compress=5)
        answers_start_positions = answers_end_positions = [cls_index]

    return answers_start_positions, answers_end_positions


# Validation split preprocessing, the features prodced from this
# is used to evaluate the metrics not the loss
def preprocess_validation_features(examples,
                                   question_column_name,
                                   context_column_name,
                                   tokenizer,
                                   max_seq_length,
                                   pad_on_right,
                                   data_args):
    # this preprocessing is shared between validation features and train features
    # (it does the tokenization process and question cleaning)
    tokenized_examples, sample_mapping = common_preprocessing(context_column_name,
                                                              data_args,
                                                              examples,
                                                              max_seq_length,
                                                              pad_on_right,
                                                              question_column_name,
                                                              tokenizer)

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["offset_mapping"])):

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        if tokenized_examples.is_fast:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (token_char_span if sequence_ids[token_id] == context_index else None)  # token_char_span is a tuple (start_char , end_char)
                for token_id, token_char_span in enumerate(tokenized_examples["offset_mapping"][i])
            ]

    return tokenized_examples
