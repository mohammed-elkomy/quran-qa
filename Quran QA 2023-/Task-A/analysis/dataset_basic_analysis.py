from matplotlib import pyplot as plt

from data_scripts import read_run_file, read_qrels_file
import pandas as pd

dev_qrel = read_qrels_file("data/QQA23_TaskA_qrels_dev.gold")
train_qrel = read_qrels_file("data/QQA23_TaskA_qrels_train.gold")
dev_dict = dev_qrel.groupby("qid").apply(lambda x: list(x["docid"])).to_dict()
train_dict = train_qrel.groupby("qid").apply(lambda x: list(x["docid"])).to_dict()

merge_dict = {}
for k in train_dict:
    merge_dict[k] = train_dict[k]
for k in dev_dict:
    merge_dict[k] = dev_dict[k]


def split_ids(data_dict):
    no_answer = [k for k, v in data_dict.items() if v == ["-1"]]
    single_answer = [k for k, v in data_dict.items() if len(v) == 1 and k not in no_answer]
    multi_answer = [k for k, v in data_dict.items() if len(v) > 1]
    return no_answer, single_answer, multi_answer


no_answer, single_answer, multi_answer = split_ids(train_dict)
print("Train")
print("Multi-answer", len(multi_answer), )
print("Single-answer", len(single_answer))
print("No Answer", len(no_answer))
no_answer, single_answer, multi_answer = split_ids(dev_dict)
print("Dev")
print("Multi-answer", len(multi_answer), )
print("Single-answer", len(single_answer))
print("No Answer", len(no_answer))
no_answer, single_answer, multi_answer = split_ids(merge_dict)

assert len(multi_answer) + len(single_answer) + len(no_answer) == 199

x_data = ["Multi-answer", "Single-answer", "No Answer"]
y_data = [len(multi_answer), len(single_answer), len(no_answer)]
# Creating the bar chart
plt.bar(x_data, y_data)
# Adding text on top of each bar
for i in range(len(x_data)):
    plt.text(x_data[i], y_data[i], f"{y_data[i] / sum(y_data):0.2f}%", ha='center', va='bottom')
# Adding labels and title
plt.xlabel('Question Type')
plt.ylabel('Number of questions', )
plt.title('Histogram of question types', )
# Displaying the chart
plt.show()

