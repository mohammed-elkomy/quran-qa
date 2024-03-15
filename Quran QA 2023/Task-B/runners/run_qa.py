import os
import sys
from collections import defaultdict
from functools import partial
from random import choices, choice

import numpy as np
from evaluate import load
from fuzzysearch import find_near_matches
from numpy.random import randint
from torch import optim
from transformers import (
    HfArgumentParser
)

sys.path.append(os.getcwd())  # for relative imports

from configs.training_args import CustomTrainingArguments
from configs.data_training_args import DataArguments
from configs.model_args import ModelArguments
from configs.utils import is_from_flax, prepare_my_output_dirs, handle_seed, config_logger, handle_last_checkpoint, logger
from data_scripts import get_raw_datasets, get_max_sequence_length, get_columns_names
from data_scripts.custom_padding_collator import get_data_collator
from data_scripts.outputs_post_processing.bert_qa_post_processing import post_processing_function
from data_scripts.trainer_preprocessing.bert_qa_preprocessing import tokenize_train_features, preprocess_validation_features, generate_training_labels
from metrics import compute_metrics
from models.utils import prepare_model, prepare_tokenizer, prepare_model_configs
from trainers.trainer_qa import QuestionAnsweringTrainer
from trainers.utils import save_to_hub, save_artifacts_compressed


def main():
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    handle_seed(training_args)
    log_level = config_logger(training_args)

    prepare_my_output_dirs(training_args)
    data_logs_file = open(os.path.join(training_args.my_output_dir, "data_logs_file.txt"), "w", encoding="utf-8")

    last_checkpoint = handle_last_checkpoint(training_args)

    raw_datasets = get_raw_datasets(data_args, model_args)

    # Load pretrained model and tokenizer
    from_flax = is_from_flax(model_args)  # is the model trained with flax
    model_config = prepare_model_configs(model_args)
    tokenizer = prepare_tokenizer(model_args)
    model = prepare_model(model_config, from_flax, model_args)
    print(model)
    # Preprocessing the datasets.
    dataset_column_names, question_column_name, context_column_name, answer_column_name = get_columns_names(raw_datasets, training_args)
    max_seq_length = get_max_sequence_length(data_args, tokenizer)
    pad_on_right = tokenizer.padding_side == "right"

    preprocess_validation_features_ = partial(preprocess_validation_features,
                                              question_column_name=question_column_name,
                                              context_column_name=context_column_name,
                                              tokenizer=tokenizer,
                                              max_seq_length=max_seq_length,
                                              pad_on_right=pad_on_right,
                                              data_args=data_args)

    # Preprocessing is slightly different for training and evaluation.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_raw_examples = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_raw_examples = train_raw_examples.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            ##############################
            # step 1: tokenization
            ##########
            tokenize_train_features_ = partial(tokenize_train_features,
                                               question_column_name=question_column_name,
                                               context_column_name=context_column_name,
                                               tokenizer=tokenizer,
                                               max_seq_length=max_seq_length,
                                               pad_on_right=pad_on_right,
                                               data_args=data_args,
                                               )

            # train_processed_dataset has no labels yet, labels are added in a different stage
            # for that reason we don't remove columns because they will be reused
            train_tokenized_dataset = train_raw_examples.map(
                tokenize_train_features_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            ##########
            # step 2: generating labels
            ##########
            generate_training_labels_ = partial(generate_training_labels,
                                                tokenizer=tokenizer,
                                                answer_column_name=answer_column_name,
                                                context_column_name=context_column_name,
                                                data_args=data_args,
                                                pad_on_right=pad_on_right,
                                                data_logs_file=data_logs_file,
                                                )

            train_processed_dataset = train_tokenized_dataset.map(
                generate_training_labels_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=dataset_column_names,
                desc="Generating new labels for train dataset",
            )
            ##############################
            # process training dataset as a validation set (useful for logging)
            train_eval_processed_dataset = train_raw_examples.map(
                preprocess_validation_features_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset for evaluation",
            )
            ##############################

        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation because of long evidence text, We select only specified max samples
            train_processed_dataset = train_processed_dataset.select(range(data_args.max_train_samples))
            train_eval_processed_dataset = train_eval_processed_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_raw_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_raw_examples = eval_raw_examples.select(range(data_args.max_eval_samples))

        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_processed_dataset = eval_raw_examples.map(
                preprocess_validation_features_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["validation"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_processed_dataset = eval_processed_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_raw_examples = raw_datasets["test"]
        test_has_no_answers = all([len(answer["text"]) == 0
                                   for answer in predict_raw_examples[answer_column_name]]
                                  )  # if all samples have no answer we can't compute metrics

        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_raw_examples = predict_raw_examples.select(range(data_args.max_predict_samples))

        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_processed_dataset = predict_raw_examples.map(
                preprocess_validation_features_,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_datasets["test"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",

            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_processed_dataset = predict_processed_dataset.select(range(data_args.max_predict_samples))

    data_collator = get_data_collator(data_args, tokenizer, training_args)

    metric_file = data_args.eval_metric

    metric = load(metric_file)

    if training_args.device.type == "cpu":
        training_args.dataloader_pin_memory = False
    # Initialize our Trainer
    compute_metrics_ = partial(compute_metrics, metric=metric, no_answer_threshold=data_args.no_answer_threshold, cutoff=data_args.metric_cutoff)
    post_processing_function_ = partial(post_processing_function,
                                        data_args=data_args,
                                        training_args=training_args,
                                        log_level=log_level,
                                        answer_column_name=answer_column_name,
                                        context_column_name=context_column_name,
                                        )
    optimizers, ema = (None, None), None
    trainer = QuestionAnsweringTrainer(
        # custom params
        ###############################################
        # to perform any evaluation we need both unprocessed examples + processed dataset
        ##### train #####
        train_raw_examples=train_raw_examples if training_args.do_train else None,  # evaluating the train dataset to compare them during training
        train_eval_dataset=train_eval_processed_dataset if training_args.do_train else None,  # evaluating the train dataset to compare them during training
        ##### eval #####
        eval_raw_examples=eval_raw_examples if training_args.do_eval else None,  # evaluation dataset
        ##### test #####
        test_raw_examples=predict_raw_examples if training_args.do_predict else None,  # evaluating the test dataset to compare them during training
        test_dataset=predict_processed_dataset if training_args.do_predict else None,
        test_has_no_answers=test_has_no_answers if training_args.do_predict else None,
        ################
        post_processing_function=post_processing_function_,
        ###############################################
        # trainer params
        model=model,
        args=training_args,
        train_dataset=train_processed_dataset if training_args.do_train else None,
        eval_dataset=eval_processed_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_,
        optimizers=optimizers,

    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_processed_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_processed_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics, data_args, training_args)
        trainer.save_state()

    # I'm not sure why early stopping requires loading the best model, I will load the last check point
    # I believe this is needed for this dataset because we observed huge variations in any evaluation split,
    # this means decisions made based on that will be biased and don't really reflect the reality
    # (when training with train+dev combined blindly without evaluation at the final phase)
    if not training_args.save_last_checkpoint_to_drive and training_args.do_train:
        # training_args.save_last_checkpoint_to_drive is only true for pretraining, for which the datasets are larger and the observed variations in evaluation will be less (The justification above doesn't hold)
        trainer.load_last_checkpoint(training_args)
        print("loaded best checkpoint")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_processed_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_processed_dataset))

        trainer.save_metrics("eval", metrics, data_args, training_args)

        unloggable_keys = [k for k, v in metrics.items() if type(v) == dict]
        for key in unloggable_keys:
            metrics.pop(key)

        trainer.log_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        trainer.predict(predict_processed_dataset, predict_raw_examples)
        # predict will call the post-processing function and dump the predictions to the disk

    save_to_hub(data_args, model_args, trainer, training_args)

    if not os.environ.get("LOCALTESTING", False):
        save_artifacts_compressed(training_args, data_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
""" 

"""
