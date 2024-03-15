from collections import defaultdict, Counter
from functools import partial

from datasets import load_dataset, DownloadConfig, concatenate_datasets
import matplotlib.pyplot as plt

qrcd_dataset = load_dataset("data_scripts/loader_scripts/qrcd_v1_2_dataset_loader.py",
                            data_files={'train': '../../data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl',
                                        'validation': '../../data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl',
                                        },
                            download_config=DownloadConfig(local_files_only=True)
                            )


def add_source(item, source):
    item["source"] = source
    return item


def get_num_answers_class(entry):
    num = len(entry["answers"]["answer_start"])
    if num == 0:
        return "no answer"
    if num == 1:
        return "single answer"
    else:
        return "multi answer"


def show_dist(collection, tag):
    t_cnt = len(collection["train"])
    v_cnt = len(collection["validation"])
    print(tag, t_cnt, v_cnt, t_cnt + v_cnt)


train_dataset = qrcd_dataset["train"]
eval_dataset = qrcd_dataset["validation"]
qrcd_dataset = concatenate_datasets([
    qrcd_dataset["train"].map(partial(add_source, source="train")),
    qrcd_dataset["validation"].map(partial(add_source, source="validation")),
])

ques = defaultdict(set)
pairs = defaultdict(list)
triplets = defaultdict(list)
num_answers = defaultdict(lambda: defaultdict(int))
answers_texts = []
for entry in qrcd_dataset:
    pairs[entry["source"]].append((entry["context"], entry["question"]))
    ques[entry["source"]].add(entry["question"])
    for text, answer_start in zip(entry["answers"]["text"], entry["answers"]["answer_start"]):
        triplets[entry["source"]].append((entry["context"], entry["question"], answer_start))
        answers_texts.append(text)
    if len(entry["answers"]["answer_start"]) == 0:
        triplets[entry["source"]].append((entry["context"], entry["question"], "NO"))
    num_answers[entry["source"]][get_num_answers_class(entry)] += 1

print(len(qrcd_dataset))
show_dist(ques, "Questions")
show_dist(pairs, "Pairs")
show_dist(triplets, "Triplets")

print("NO answer ratio to pairs")
print(100 * num_answers["validation"]["no answer"] / sum(num_answers["validation"].values()))
print(100 * num_answers["train"]["no answer"] / sum(num_answers["train"].values()))
print(
    100 * (num_answers["train"]["no answer"] + num_answers["validation"]["no answer"]) / (sum(num_answers["train"].values()) + sum(num_answers["validation"].values()))
)

import matplotlib.pyplot as plt

# Draw histogram
plt.hist([len(a.split()) for a in answers_texts], bins=20)
plt.xlabel('Answer length')
plt.ylabel('Frequency')
plt.title('Histogram of answers length')
plt.show()


def get_per_type(dataset):
    question_to_type = {}
    question_to_num_sample = defaultdict(int)
    for q, group_df in dataset.to_pandas().groupby("question"):
        answer_counts = [answer["text"].shape[0] for answer in group_df["answers"]]
        samples_count = group_df["answers"].shape[0]

        if 0 in answer_counts:
            # zero only occurs and the question is not answerable from anywhere in the quran
            assert sum(answer_counts) == 0
        # question_to_num_answers[q] = len(answer_counts) if sum(answer_counts) == 0 else sum(answer_counts) # verify triplet counts
        if sum(answer_counts) == 0:
            question_to_type[q] = "no answer"
        elif sum(answer_counts) == 1:
            question_to_type[q] = "single answer"
        else:
            question_to_type[q] = "multi answer"

        for answer_count in answer_counts:
            if answer_count == 0:
                question_to_num_sample["no answer"] += 1
            elif answer_count == 1:
                question_to_num_sample["single answer"] += 1
            elif answer_count > 1:
                question_to_num_sample["multi answer"] += 1

    assert sum(question_to_num_sample.values()) in [1155, 992, 163], sum(question_to_num_sample.values())
    return question_to_type, question_to_num_sample


question_to_type, question_to_num_sample = get_per_type(qrcd_dataset)
t_question_to_type, t_question_to_num_sample = get_per_type(train_dataset)
e_question_to_type, e_question_to_num_sample = get_per_type(eval_dataset)


def draw_bar_perc(x_label, y_label, title, x_data, y_data):
    # Creating the bar chart
    plt.bar(x_data, y_data)
    # Adding text on top of each bar
    for i in range(len(x_data)):
        plt.text(x_data[i], y_data[i], f"{y_data[i] / sum(y_data):0.2f}%", ha='center', va='bottom')
    # Adding labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # Displaying the chart
    plt.show()


# figure 1: dist of question types

question_type = list(Counter(question_to_type.values()).keys())
counts_per_type = list(Counter(question_to_type.values()).values())

draw_bar_perc(x_label='Question Type',
              y_label='Number of questions',
              title='Histogram of question types',
              x_data=question_type,
              y_data=counts_per_type)

# figure 2: dist of pairs per question type

draw_bar_perc(x_label='Question Type',
              y_label='Number of pairs',
              title='Distribution of pairs per question type',
              x_data=list(e_question_to_num_sample.keys()),
              y_data=list(e_question_to_num_sample.values()))

print("all", question_to_num_sample)
print("train", t_question_to_num_sample)
print("eval", e_question_to_num_sample)
