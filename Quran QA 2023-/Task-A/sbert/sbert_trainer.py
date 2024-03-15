import os
import random
import sys

import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

sys.path.append(os.getcwd())  # for relative imports

from configs.data_training_args import DataArguments
from configs.model_args import ModelArguments
from configs.training_args import CustomTrainingArguments
from configs.utils import handle_seed
from cross_encoder import make_training_triplets, make_dev_triplets, save_model_to_drive, prepare_my_output_dirs, config_logger, zip_inference_data
from sbert.inference import infer_relevance
from data_scripts import read_docs_file, read_query_file, read_qrels_file, read_run_file
from metrics.Custom_TaskA_eval import evaluate_task_a

parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

log_level = config_logger(training_args)
handle_seed(training_args)
training_args.output_dir = "bi-" + training_args.output_dir
prepare_my_output_dirs(training_args)
doc_df = read_docs_file(data_args.doc_file)

if training_args.do_train:
    train_qrel_df = read_qrels_file(data_args.train_qrel_file, )
    train_query_df = read_query_file(data_args.train_query_file, )
    if training_args.save_last_checkpoint_to_drive:
        samples_per_query = 2  # pretraining data from TYDI-QA
    else:
        samples_per_query = 10
    print("samples_per_query", samples_per_query)
    training_triplets = make_training_triplets(train_qrel_df, train_query_df, doc_df, samples_per_query=samples_per_query)

if training_args.do_eval:
    dev_qrel_df = read_qrels_file(data_args.validation_qrel_file, )
    dev_query_df = read_query_file(data_args.validation_query_file, )
    dev_triplets = make_dev_triplets(dev_qrel_df, dev_query_df, doc_df)
    dev_infer_data = dev_query_df, doc_df

    if data_args.max_eval_samples is not None:
        dev_triplets = random.choices(dev_triplets, k=data_args.max_eval_samples)

    dev_queries = [q_text for q_text, _, _ in dev_triplets]
    dev_docs = [doc_text for _, doc_text, _ in dev_triplets]
    dev_labels = [float(label) for _, _, label in dev_triplets]


if training_args.do_predict:
    test_query_df = read_query_file(data_args.test_query_file, )
    test_infer_data = test_query_df, doc_df
    # no qrel file for testing data

train_triplets = [InputExample(texts=(q_text, doc_text), label=float(label)) for q_text, doc_text, label in training_triplets]

if data_args.max_train_samples is not None:
    # During Feature creation dataset samples might increase, we will select required samples again
    train_triplets = random.choices(train_triplets, k=data_args.max_train_samples)




train_dataloader = DataLoader(train_triplets, shuffle=True, batch_size=training_args.per_device_train_batch_size)
model = SentenceTransformer(model_args.model_name_or_path, )

train_loss = losses.CosineSimilarityLoss(model)

warmup_steps = 1000


# demo_rerank(model)

def eval_dataset(inference_data):
    dev_infer_df = infer_relevance(model, *inference_data, tok_k_relevant=data_args.tok_k_relevant)
    results = evaluate_task_a(dev_qrel_df, dev_infer_df)
    print(results[0]["overall"])


def eval_demo():
    if not training_args.save_last_checkpoint_to_drive:
        print("dev_infer_data")
        eval_dataset(dev_infer_data)


def epoch_callback(score, epoch, steps):
    print(score, epoch, steps)
    if epoch % 5:
        eval_demo()


eval_demo()

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=int(training_args.num_train_epochs),
          evaluator=EmbeddingSimilarityEvaluator(dev_queries, dev_docs, dev_labels) if training_args.do_eval else None,
          warmup_steps=warmup_steps,
          save_best_model=True,
          callback=epoch_callback,
          output_path=os.path.join(training_args.output_dir, "checkpoints", "-" + str(training_args.seed)),
          optimizer_params={'lr': training_args.learning_rate},
          show_progress_bar=False)

model.save(os.path.join(training_args.output_dir, f"last-checkpoint"))

if training_args.save_last_checkpoint_to_drive:
    save_model_to_drive(training_args)
else:
    dev_infer_df = infer_relevance(model, *dev_infer_data, tok_k_relevant=data_args.tok_k_relevant)
    test_infer_df = infer_relevance(model, *test_infer_data, tok_k_relevant=data_args.tok_k_relevant)
    dev_infer_df.to_csv(os.path.join(training_args.my_output_dir, "eval_inference.tsv"), sep="\t", index=False, header=False)
    test_infer_df.to_csv(os.path.join(training_args.my_output_dir, "test_inference.tsv"), sep="\t", index=False, header=False)
    zip_inference_data(training_args, data_args)
