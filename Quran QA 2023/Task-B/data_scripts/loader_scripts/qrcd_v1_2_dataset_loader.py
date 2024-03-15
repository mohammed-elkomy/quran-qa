"""QRCD v1.2: Qur'anic Reading Comprehension Dataset."""

import datasets
from datasets.tasks import QuestionAnsweringExtractive



logger = datasets.logging.get_logger(__name__)

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


class QRCDConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for QRCD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QRCDConfig, self).__init__(**kwargs)


class QRCD(datasets.GeneratorBasedBuilder):
    """QRCD: Version 1.2."""

    BUILDER_CONFIGS = [
        QRCDConfig(
            name="plain_text",
            version=datasets.Version("1.2.0", ""),
            description="QRCD v1.2",

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
            homepage="https://sites.google.com/view/quran-qa-2023/task-b/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        # self.config.data_files
        print(self.config.data_files)
        dataset_files = dl_manager.download_and_extract(self.config.data_files)
        print("dataset_files", dataset_files)  # a list is returned

        found_data_files = [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dataset_files["train"][0], }),
        ]

        if "validation" in dataset_files:
            found_data_files.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dataset_files["validation"][0]}), )

        # test dataset with answers
        if "test" in dataset_files:
            found_data_files.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": dataset_files["test"][0]}), )

        # test dataset without answers
        if "no_answer" in dataset_files:
            found_data_files.append(datasets.SplitGenerator(name="no_answer", gen_kwargs={"filepath": dataset_files["no_answer"][0]}), )

        return found_data_files

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        passage_question_objects = read_JSONL_file(filepath)
        print(f"\n\n\nLOADED {filepath}, with {len(passage_question_objects)} samples \n\n\n")

        for key, pq_object in enumerate(passage_question_objects):
            surah = pq_object.surah
            verses = pq_object.verses
            title = f"surah:{surah}, verses:{verses}"
            context = pq_object.passage
            question = pq_object.question
            pq_id = pq_object.pq_id

            answer_starts = [answer.start_char for answer in pq_object.answers]
            answers = [answer.text for answer in pq_object.answers]
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
    from datasets import load_dataset, DownloadConfig

    dataset = load_dataset(__file__,
                           data_files={'train': '../../data/QQA23_TaskB_qrcd_v1.2_train_preprocessed.jsonl',
                                       'validation': '../../data/QQA23_TaskB_qrcd_v1.2_dev_preprocessed.jsonl',
                                       },
                           download_config=DownloadConfig(local_files_only=True)
                           )
    dataset["train"].to_pandas().to_excel('../../artifacts/qrcd_v1.2_train.xlsx')
    dataset["validation"].to_pandas().to_excel('../../artifacts/qrcd_v1.2_dev.xlsx')

    exit()
from .read_write_qrcd import read_JSONL_file