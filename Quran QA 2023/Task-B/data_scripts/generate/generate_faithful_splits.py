"""
this file was taken from https://github.com/AMR-KELEG/SMASH-QuranQA and adapted for my analysis
"""
import itertools
import json
import random
from disjoint_set import DisjointSet
import numpy as np
import pandas as pd
from tabulate import tabulate

DEV_PERCENTAGE = 13.3


def load_datafile(filename, split):
    with open(filename, "r") as f:
        data = [json.loads(l) for l in f]
    for s in data:
        s["split"] = split
    return data


def get_normal_split(split_df, dev_percentage, random_state):
    no_dev_samples = round((dev_percentage / 100) * split_df.shape[0])
    dev_split_df = split_df.sample(n=no_dev_samples, random_state=random_state)
    train_split_df = split_df.drop(index=dev_split_df.index)
    return train_split_df, dev_split_df


def dump_to_file(split_df, split_type, filename):
    keys_to_dump = ['pq_id', 'passage', 'surah', 'verses', 'question', 'answers', 'split']
    with open(f"data/{filename}_{split_type}.jsonl", "w") as f:
        for l in split_df.to_dict("records"):
            l = {key: l[key] for key in keys_to_dump}
            f.write(json.dumps(l, ensure_ascii=False))
            f.write("\n")


def get_leakage_groups(source_df):
    """
    uses disjoint set to find all leakage clusters (connected components)
    :param source_df:
    :return:
    """
    ds = DisjointSet()
    for (x_idx, x_entry), (y_idx, y_entry) in itertools.product(source_df.iterrows(), source_df.iterrows()):
        if x_idx != y_idx and (x_entry["passage"] == y_entry["passage"] or x_entry["question"] == y_entry["question"]):
            # connect them
            ds.union(x_idx, y_idx)

    leakage_dfs = [source_df.loc[list(group)] for group in list(ds.itersets())]

    return leakage_dfs


def sample_from_groups(indomain_dfs_no_answer, limit=6):
    sampled_dfs = []
    total_sampled = 0
    while total_sampled < limit:
        sampled_idx = random.randint(0, len(indomain_dfs_no_answer) - 1)
        if indomain_dfs_no_answer[sampled_idx].shape[0] < limit - total_sampled + 2 :
            sampled_dfs.append(indomain_dfs_no_answer[sampled_idx])
            total_sampled += indomain_dfs_no_answer[sampled_idx].shape[0]
            indomain_dfs_no_answer.pop(sampled_idx)
    return sampled_dfs


def generate_faithful_splits_d2_d3_only_dev(leakage_indomain_df, non_leakage_indomain_df, easy_ood_df, hard_ood_df, random_state, processed=False):
    # I will change the methodology introduced by AMR by not splitting D1 and D4 into train-dev.
    # This means:
    # D1_in_leakage > only for training (because it's trivial for testing)
    # D4_ood_easy > only for training (because it's trivial for testing)
    # D3_ood_hard > only for dev (rare questions)
    # D2_in_no_leakage > split them into (overlapping passages), to show confusing repeated paragraphs in both training and dev to measure the generalization abilities

    filenames = [
        "D1_my_in_leakage",
        "D4_my_ood_easy",
        "D2_my_in_no_leakage",
        "D3_my_ood_hard",
    ]
    dfs = [
        leakage_indomain_df,
        easy_ood_df,
        non_leakage_indomain_df,
        hard_ood_df,
    ]

    train_split_dfs = []
    eval_split_dfs = []
    ##############################
    # D1_in_leakage: in domain leakage
    # take some no-answer leakage groups (to balance the no-answer case among train / dev splits)
    leakage_indomain_df_no_answer = leakage_indomain_df[leakage_indomain_df["answer"] == ""]
    leakage_indomain_df_has_answer = leakage_indomain_df[leakage_indomain_df["answer"] != ""]
    del leakage_indomain_df
    indomain_dfs_no_answer = get_leakage_groups(leakage_indomain_df_no_answer)
    sampled_indomain_leakage_dfs = sample_from_groups(indomain_dfs_no_answer, limit=6)
    t_leakage_indomain = shuffle_df(pd.concat(indomain_dfs_no_answer + [leakage_indomain_df_has_answer]), random_state)  # indomain_dfs_no_answer is updated here by sample_from_groups
    # leakage_groups = list(leakage_indomain_df.groupby("question"))
    train_split_dfs.append(t_leakage_indomain)
    dump_to_file(t_leakage_indomain, "train", filenames[0])

    d_leakage_indomain = shuffle_df(pd.concat(sampled_indomain_leakage_dfs), random_state)  # indomain_dfs_no_answer is updated here by sample_from_groups
    eval_split_dfs.append(d_leakage_indomain)
    dump_to_file(d_leakage_indomain, "dev", filenames[0])
    assert set(pd.concat(indomain_dfs_no_answer)["passage"]).isdisjoint(set(d_leakage_indomain["passage"]))
    assert set(pd.concat(indomain_dfs_no_answer)["question"]).isdisjoint(set(d_leakage_indomain["question"]))
    ##############################
    # D4_my_ood_easy: repeated questions are easier because of the lexical similarity of answers
    train_split_dfs.append(easy_ood_df)
    dump_to_file(easy_ood_df, "train", filenames[1])
    ##############################
    # - `non_leakage_indomain_df`: Repeated passages with different questions and answers
    # take one sample with repeated passages for development and the rest for training
    non_leakage_indomain_df = shuffle_df(non_leakage_indomain_df, random_state)
    d_split_non_leakage = non_leakage_indomain_df.drop_duplicates(subset=["passage"])
    # Use all remaining repeated questions in the training dataset
    t_split_non_leakage = non_leakage_indomain_df.drop(index=d_split_non_leakage.index)
    ##################
    # to balance the numbers (closer to the original ratio),
    # this means the training set may include a repeated passage with different answers (a confusing example)
    new_d_split = d_split_non_leakage.sample(n=95, random_state=random_state)
    new_t_split = pd.concat([t_split_non_leakage, d_split_non_leakage.drop(index=new_d_split.index)])
    d_split_non_leakage, t_split_non_leakage = new_d_split, new_t_split
    ###################
    train_split_dfs.append(t_split_non_leakage)
    eval_split_dfs.append(d_split_non_leakage)

    dump_to_file(t_split_non_leakage, "train", f"{filenames[2] + ('-processed' if processed else '')}-{random_state}")
    dump_to_file(d_split_non_leakage, "dev", f"{filenames[2] + ('-processed' if processed else '')}-{random_state}")
    ##############################
    # ood hard
    ood_hard = dfs[-1]
    dump_to_file(
        ood_hard.drop(["passage_answer", "question_answer"], axis=1),
        "dev",
        filenames[-1] + ('-processed' if processed else ''),
    )
    eval_split_dfs.append(ood_hard)
    ##############################
    faithful_train = pd.concat(train_split_dfs)
    faithful_dev = pd.concat(eval_split_dfs)
    assert faithful_train.merge(faithful_dev, on="pq_id").shape[0] == 0  # disjoint

    dump_to_file(faithful_train, "train", f"my-faithful-{('processed-' if processed else '')}{random_state}")
    dump_to_file(faithful_dev, "dev", f"my-faithful-{('processed-' if processed else '')}{random_state}")

    print(random_state)

    table = tabulate(
        [
            ["Train", "Dev"],
            ["leakage", no_count_format(t_leakage_indomain), no_count_format(d_leakage_indomain)],
            ["non_leakage", no_count_format(t_split_non_leakage), no_count_format(d_split_non_leakage)],
            ["ood_hard", 0, no_count_format(ood_hard)],
            ["ood_easy", no_count_format(easy_ood_df), 0],
            ["total", no_count_format(faithful_train), no_count_format(faithful_dev)],
            ["No answer %", round(np.mean(faithful_train["answer"] == "") * 100, 4),
             round(np.mean(faithful_dev["answer"] == "") * 100, 4)],
        ], headers="firstrow",
        tablefmt="fancy_grid",
        stralign="center",
        numalign="center")

    print(table)


def no_count_format(df):
    no_count = sum(df["answer"] == "")
    total_count = df.shape[0]
    return f"{total_count} ({no_count})"


def generate_faithful_splits(leakage_indomain_df, non_leakage_indomain_df, easy_ood_df, hard_ood_df, random_state):
    # this strategy is proposed by AMR KELEG, my proposal is implemented in generate_my_faithful_splits

    filenames = [
        "D1_in_leakage",
        "D4_ood_easy",
        "D2_in_no_leakage",
        "D3_ood_hard",
    ]
    dfs = [
        leakage_indomain_df,
        easy_ood_df,
        non_leakage_indomain_df,
        hard_ood_df,
    ]

    train_split_dfs = []
    eval_split_dfs = []
    ##############################
    # in domain leakage
    ###########
    # Show model one version of a repeated question to avoid overfitting!
    t_split_leakage = leakage_indomain_df.drop_duplicates(subset=["question_answer"]).drop_duplicates(subset=["passage_answer"])
    # Use all remaining repeated questions in the development dataset
    d_split_leakage = leakage_indomain_df.drop(index=t_split_leakage.index)
    train_split_dfs.append(t_split_leakage)
    eval_split_dfs.append(d_split_leakage)

    dump_to_file(t_split_leakage, "train", filenames[0])
    dump_to_file(d_split_leakage, "dev", filenames[0])
    ##############################
    # ood easy + no leakage
    ###########
    for typed_df, filename in zip(dfs[1:-1], filenames[1:-1]):
        t_split, d_split = get_normal_split(
            typed_df.drop(["passage_answer", "question_answer"], axis=1), DEV_PERCENTAGE, random_state
        )
        dump_to_file(t_split, "train", f"{filename}-{random_state}")
        dump_to_file(d_split, "dev", f"{filename}-{random_state}")
        train_split_dfs.append(t_split)
        eval_split_dfs.append(d_split)
    ##############################
    # ood hard
    ood_hard = dfs[-1]
    dump_to_file(
        ood_hard.drop(["passage_answer", "question_answer"], axis=1),
        "dev",
        filenames[-1],
    )
    eval_split_dfs.append(ood_hard)
    ##############################
    assert pd.concat(train_split_dfs).merge(pd.concat(eval_split_dfs), on="pq_id").shape[0] == 0  # disjoint

    dump_to_file(pd.concat(train_split_dfs), "train", f"faithful-{random_state}")
    dump_to_file(pd.concat(eval_split_dfs), "dev", f"faithful-{random_state}")


def rebalance(t_split, d_split, random_state):
    """
    takes back some sample from d_split into t_split to make the relative sizes closer to the original ratio
    """
    required_dev_size = int(round((t_split.shape[0] + d_split.shape[0]) * DEV_PERCENTAGE / 100))
    new_d_split = d_split.sample(n=required_dev_size, random_state=random_state)

    new_t_split = pd.concat([t_split, d_split.drop(index=new_d_split.index)])
    assert t_split.merge(d_split, on="pq_id").shape[0] == 0  # disjoint
    assert new_d_split.merge(new_t_split, on="pq_id").shape[0] == 0  # disjoint
    return new_t_split, new_d_split


def shuffle_df(df, random_state):
    return df.sample(frac=1, random_state=random_state).reset_index()


def categorize_qrcd_dataset(train_data, eval_data):
    concat_data = train_data + eval_data

    df = pd.DataFrame(concat_data)
    df["answer"] = df["answers"].apply(
        lambda a_list: "|".join(
            [a["text"] for a in sorted(a_list, key=lambda l: l["start_char"])]
        )
    )
    df["passage_answer"] = df.apply(
        lambda row: f"{row['passage']}|{row['answer']}", axis=1
    )
    df["question_answer"] = df.apply(
        lambda row: f"{row['question']}|{row['answer']}", axis=1
    )
    questions_count_dict = {
        row["index"]: row["question"]
        for i, row in df["question"].value_counts().reset_index().iterrows()
    }
    print(len(eval_data) / (len(train_data) + len(eval_data)))
    # Having same (question or passage) & answer pairs repeated
    leakage_indomain_df = df[
        (df["passage_answer"].duplicated(keep=False))
        | (df["question_answer"].duplicated(keep=False))
        ].sort_values(by="passage_answer")
    leakage_indomain_df.shape[0], \
        leakage_indomain_df.shape[0] / df.shape[0], \
        len(leakage_indomain_df["passage"].unique()), \
        len(leakage_indomain_df["question"].unique())
    unique_passage_df = df.drop(index=leakage_indomain_df.index)
    unique_passage_df = unique_passage_df[
        ~unique_passage_df["passage"].duplicated(keep=False)
    ]
    print(unique_passage_df.shape)
    # TODO: Find questions that aren't part of the unique passage_df
    threshold = 3
    total_ood_df = unique_passage_df[
        unique_passage_df["question"].apply(lambda q: questions_count_dict[q])
        <= threshold
        ]
    print(total_ood_df.shape)
    context_ood_df = unique_passage_df[
        unique_passage_df["question"].apply(lambda q: questions_count_dict[q])
        > threshold
        ]
    print(context_ood_df.shape)
    non_leakage_indomain_df = df.drop(
        index=leakage_indomain_df.index.tolist() + unique_passage_df.index.tolist()
    )
    assert (
            leakage_indomain_df.shape[0]
            + non_leakage_indomain_df.shape[0]
            + total_ood_df.shape[0]
            + context_ood_df.shape[0]
            == df.shape[0]
    )
    return leakage_indomain_df, non_leakage_indomain_df, context_ood_df, total_ood_df


def main():
    train_data = load_datafile("data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl", split="train")
    eval_data = load_datafile("data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl", split="dev")

    leakage_indomain_df, non_leakage_indomain_df, easy_ood_df, hard_ood_df = categorize_qrcd_dataset(train_data, eval_data)

    # # Types of datasets
    # - `leakage_indomain_df`: Multiple questions/passages having the same answer for paraphrased questions or different similar passages
    # - `hard_ood_df`: Hard rare questions
    # - `easy_ood_df`: Repeated questions on new passages
    # - `non_leakage_indomain_df`: Repeated passages with different questions and answers

    for i in range(0, 100):
        random.seed(i)
        generate_faithful_splits_d2_d3_only_dev(leakage_indomain_df, non_leakage_indomain_df, easy_ood_df, hard_ood_df, random_state=i, processed=True)


if __name__ == "__main__":
    # TODO: make sure the files have the same hashes on all machines
    # md5sum * | sort -k 2

    print("make sure to run this script from the repo root directory, 'data_scripts/generate/generate_faithful_splits.py'")
    main()

