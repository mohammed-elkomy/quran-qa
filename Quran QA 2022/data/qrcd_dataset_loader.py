# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""SQUAD: The Stanford Question Answering Dataset."""
from collections import defaultdict

import datasets
from datasets import DownloadConfig, load_dataset
from datasets.tasks import QuestionAnsweringExtractive
from matplotlib import pyplot as plt, pylab

logger = datasets.logging.get_logger(__name__)

import json


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


class Answer:
    def __init__(self, dictionary) -> None:
        self.text = dictionary["text"]
        self.start_char = dictionary["start_char"]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start_char": self.start_char
        }  # answer_dict


class PassageQuestion:
    def __init__(self, dictionary, include_answers) -> None:

        self.pq_id = dictionary["pq_id"]
        self.passage = dictionary["passage"]
        self.surah = dictionary["surah"]
        self.verses = dictionary["verses"]
        self.question = dictionary["question"]
        if include_answers:
            self.answers = []
            for answer in dictionary["answers"]:
                self.answers.append(Answer(answer))

    def to_dict(self, include_answers=True) -> dict:
        passage_question_dict = {
            "pq_id": self.pq_id,
            "surah": self.surah, "verses": self.verses,
            "passage": self.passage,
            "question": self.question,

        }  # passage_question_dict
        if include_answers:
            passage_question_dict["answers"] = [x.to_dict() for x in self.answers]

        return passage_question_dict


def read_JSONL_file(file_path, has_answers) -> list:
    data_in_file = load_jsonl(file_path)

    # get list of PassageQuestion objects
    passage_question_objects = []
    for passage_question_dict in data_in_file:
        # instantiate a PassageQuestion object
        pq_object = PassageQuestion(passage_question_dict, has_answers)
        passage_question_objects.append(pq_object)

    print(f"Collected {len(passage_question_objects)} Object from {file_path}")
    return passage_question_objects


_CITATION = """\
_CITATION
"""

_DESCRIPTION = """\
QRCD
"""


class QRCDConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, preprocessor, **kwargs):
        """BuilderConfig for QRCD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QRCDConfig, self).__init__(**kwargs)
        self.preprocessor = preprocessor


class QRCD(datasets.GeneratorBasedBuilder):
    """QRCD: Version 1.1."""

    BUILDER_CONFIGS = [
        QRCDConfig(
            preprocessor=None,
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",

        ),
    ]

    DEFAULT_CONFIG_NAME = "plain_text"
    BUILDER_CONFIG_CLASS = QRCDConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        # self.config.data_files
        preprocessor = self.config.preprocessor
        dataset_files = dl_manager.download_and_extract(self.config.data_files)
        print("dataset_files", dataset_files)  # a list is returned

        found_data_files = [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dataset_files["train"][0], "preprocessor": preprocessor}),
        ]

        if "validation" in dataset_files:
            found_data_files.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dataset_files["validation"][0], "preprocessor": preprocessor}),)

        if "test" in dataset_files:
            found_data_files.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": dataset_files["test"][0], "preprocessor": preprocessor, "has_answers": False}), )

        return found_data_files

    def _generate_examples(self, filepath, preprocessor, has_answers=True):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        passage_question_objects = read_JSONL_file(filepath, has_answers)
        print(f"\n\n\nLOADED {filepath}, with {len(passage_question_objects)} samples \n\n\n")
        # preprocessor = ArabertPreprocessor(model_name=preprocessor)
        for key, pq_object in enumerate(passage_question_objects):
            surah = pq_object.surah
            verses = pq_object.verses
            title = f"surah:{surah}, verses:{verses}"
            context = pq_object.passage
            question = pq_object.question
            pq_id = pq_object.pq_id

            if has_answers:  # eval dataset
                answer_starts = [answer.start_char for answer in pq_object.answers]
                answers = [answer.text for answer in pq_object.answers]
            else:  # test dataset
                answer_starts = None,
                answers = None,

            yield key, {
                "title": title,
                "context": context,
                "question": question,
                "id": pq_id,
                "answers": {
                    "answer_start": answer_starts,
                    "text": answers,
                },
            }


if __name__ == "__main__":
    from pipe import where
    from preprocess import ArabertPreprocessor

    dataset = load_dataset(__file__,
                           data_files={'train': 'qrcd/qrcd_v1.1_train.jsonl',
                                       'validation': 'qrcd/qrcd_v1.1_dev.jsonl',
                                       'test': 'qrcd/qrcd_v1.1_test_noAnswers.jsonl',  # has no answers
                                       },
                           download_config=DownloadConfig(local_files_only=True), preprocessor=None
                           )
    dataset["train"].to_pandas().to_excel('qrcd/qrcd_v1.1_train.xlsx')
    dataset["validation"].to_pandas().to_excel('qrcd/qrcd_v1.1_dev.xlsx')
    dataset["test"].to_pandas().to_excel('qrcd/qrcd_v1.1_my_test.xlsx')

    questions =  defaultdict(set)
    for split in dataset:
        for sample in dataset[split]:
            questions[split].add( sample["question"])

    set(dataset["train"]["question"])
    set(dataset["validation"]["question"])
    set(dataset["test"]["question"])

    answer_length = defaultdict(list)
    for split in dataset:
        for sample in dataset[split]:
            q_type = sample["question"].split()[0]
            answer_length[q_type].extend([len(answer.split()) for answer in sample["answers"]["text"]])
    print()


    def split_answers(examples):
        examples["answer_max_lengths"] = []
        for example_answers in examples["answers"]:
            examples["answer_max_lengths"].append(
                max(len(answer.split()) for answer in example_answers["text"])
            )

        examples["context_length"] = [len(example_context.split()) for example_context in examples["context"]]
        examples["question_length"] = [len(example_question.split()) for example_question in examples["question"]]

        return examples


    for split in dataset:
        mapped = (dataset[split].map(
            split_answers,
            batched=True,
            num_proc=4,
            remove_columns=dataset[split].column_names,
            desc="Running tokenizer on train dataset",
        ))
        print(list(mapped["answer_max_lengths"] | where(lambda item: item > 35)))

        plt.figure()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('answer_max_lengths')
        plt.hist(mapped["answer_max_lengths"])

        plt.figure()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('context_length')
        plt.hist(mapped["context_length"])

        plt.figure()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('question_length')
        plt.hist(mapped["question_length"])
        plt.show()
