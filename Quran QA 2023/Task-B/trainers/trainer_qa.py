# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, Union, Any

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, is_torch_tpu_available, EarlyStoppingCallback, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint

from configs.utils import logger

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args,
                 train_raw_examples=None,
                 train_eval_dataset=None,
                 eval_raw_examples=None,
                 test_raw_examples=None,
                 test_dataset=None,
                 test_has_no_answers=None,
                 post_processing_function=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_raw_examples = eval_raw_examples  # raw eval dataset from loader

        self.train_raw_examples = train_raw_examples  # raw train dataset from loader
        self.train_eval_dataset = train_eval_dataset  # processed train dataset for evaluation

        self.test_raw_examples = test_raw_examples  # raw test dataset from loader
        self.test_dataset = test_dataset  # processed test dataset for evaluation
        self.test_has_no_answers = test_has_no_answers  # we can't compute metrics here

        self.post_processing_function = post_processing_function

        self.remove_callback(TensorBoardCallback)
        self.add_callback(MetricCombinedTensorBoardCallback(tb_writer=SummaryWriter(os.path.join(self.args.my_output_dir, "log"))))
        if not self.args.disable_early_stopping:
            self.add_callback(EarlyStoppingCallback(early_stopping_patience=self.args.early_stopping_patience,
                                                    early_stopping_threshold=self.args.early_stopping_threshold))

        # self.add_callback(LabelRefreshCallback())
        self.temp_err = None
        self.temp_out = None
        self.temp_compute_metrics = None


    def evaluate_train_metrics(self, ignore_keys):
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop  # use_legacy_prediction_loop defaults on false

        # train data
        train_eval_dataset = self.train_eval_dataset
        train_dataloader = self.get_eval_dataloader(train_eval_dataset)
        train_raw_examples = self.train_raw_examples
        do_eval_train = train_eval_dataset is not None and train_raw_examples is not None
        try:
            self.prepare_evaluate_routine()
            if self.args.report_train_metrics_to_tfboard and do_eval_train:
                output_train = eval_loop(
                    train_dataloader,
                    description="Train-Evaluation",
                    # No point gathering the predictions if there are no metrics, otherwise we defer to
                    # self.args.prediction_loss_only
                    prediction_loss_only=True if self.temp_compute_metrics is None else None,
                    ignore_keys=ignore_keys,
                )
        finally:
            self.restore_evaluate_routine()

        # evaluating metrics for train data
        if self.post_processing_function is not None and self.compute_metrics is not None and self.args.report_train_metrics_to_tfboard and do_eval_train:
            train_preds = self.post_processing_function(train_raw_examples, train_eval_dataset, output_train.predictions, stage="train-eval")
            return self.compute_metrics(train_preds)
        else:
            return {}

    def evaluate_eval_metrics(self, ignore_keys):
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop  # use_legacy_prediction_loop defaults on false

        # evaluation data
        eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_raw_examples = self.eval_raw_examples
        do_eval = eval_dataset is not None and eval_raw_examples is not None

        try:
            self.prepare_evaluate_routine()
            if do_eval:
                output_eval = eval_loop(
                    eval_dataloader,
                    description="Evaluation",
                    prediction_loss_only=True if self.temp_compute_metrics is None else None,
                    ignore_keys=ignore_keys,
                )
        finally:
            self.restore_evaluate_routine()

        # evaluating metrics for eval data
        if self.post_processing_function is not None and self.compute_metrics is not None and do_eval:
            eval_preds = self.post_processing_function(eval_raw_examples, eval_dataset, output_eval.predictions, stage="eval")
            return self.compute_metrics(eval_preds)
        else:
            return {}

    def evaluate_test_metrics(self, ignore_keys):
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop  # use_legacy_prediction_loop defaults on false

        # test data
        test_dataset = self.test_dataset
        test_dataloader = self.get_eval_dataloader(test_dataset)
        test_raw_examples = self.test_raw_examples
        do_test = test_dataset is not None and test_raw_examples is not None

        try:
            self.prepare_evaluate_routine()
            if do_test and not self.test_has_no_answers:
                output_test = eval_loop(
                    test_dataloader,
                    description="test",
                    prediction_loss_only=True if self.temp_compute_metrics is None else None,
                    ignore_keys=ignore_keys,
                )
        finally:
            self.restore_evaluate_routine()

        # evaluating metrics for test data
        if self.post_processing_function is not None and self.compute_metrics is not None and not self.test_has_no_answers and do_test:
            test_preds = self.post_processing_function(test_raw_examples, test_dataset, output_test.predictions, stage="test")
            return self.compute_metrics(test_preds)
        else:
            return {}

    def evaluate(self, ignore_keys=None, ):
        metrics_train = self.evaluate_train_metrics(ignore_keys)  # evaluate metrics for train data formatted as validation (to view it on tfboard)
        metrics_eval = self.evaluate_eval_metrics(ignore_keys)  # evaluate metrics for validation data
        metrics_test = self.evaluate_test_metrics(ignore_keys)  # evaluate metrics for test data if labels exist

        metrics = self.rename_metrics(metrics_train, metrics_eval, metrics_test)
        self.log(metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def prepare_evaluate_routine(self):
        err = sys.stderr
        out = sys.stdout
        sys.stderr = open(os.devnull, 'w')  # redirect those annoying logs
        sys.stdout = open(os.devnull, 'w')
        self.temp_err = err
        self.temp_out = out
        # Temporarily disable metric computation, we will do it in the loop here.
        self.temp_compute_metrics = self.compute_metrics
        self.compute_metrics = None  # if self.compute_metrics is not none, evaulate loop will call it

    def restore_evaluate_routine(self):
        sys.stderr = self.temp_err
        sys.stdout = self.temp_out
        self.compute_metrics = self.temp_compute_metrics

    @staticmethod
    def rename_metrics(metrics_train, metrics_eval, metrics_test):
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics_eval.keys()):
            if not key.startswith(f"eval_"):
                metrics_eval[f"eval_{key}"] = metrics_eval.pop(key)

        for key in list(metrics_train.keys()):
            if not key.startswith(f"train_"):
                metrics_train[f"train_{key}"] = metrics_train.pop(key)

        for key in list(metrics_test.keys()):
            if not key.startswith(f"test_"):
                metrics_test[f"test_{key}"] = metrics_test.pop(key)

        metrics = {}
        for current in [metrics_eval, metrics_train, metrics_test]:
            for key in current:
                metrics[key] = current[key]
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, ):
        """
        I changed it a bit, predict for me means you don't have access to the labels
        :param predict_dataset: after preprocessing as a dataset
        :param predict_examples: The non-preprocessed dataset, raw dataset
        :param ignore_keys:
        :return: nothing, but the predictions will be saved to the disk as the function returns
        """
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_processing_function is None or self.compute_metrics is None:
            return output

        return self.post_processing_function(predict_examples, predict_dataset, output.predictions, stage="predict")

    def save_metrics(self, split, metrics, data_args, training_args, combined=True):
        if not self.is_world_process_zero():
            return
        path = os.path.join(self.args.my_output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.my_output_dir, f"{training_args.seed}-combined.json")

            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            all_metrics["seed"] = training_args.seed
            all_metrics["best_model"] = self.state.best_model_checkpoint
            all_metrics["total_steps"] = self.state.global_step

            all_metrics["train_file"] = data_args.train_file
            all_metrics["validation_file"] = data_args.validation_file
            all_metrics["test_file"] = data_args.test_file

            all_metrics["training_args"] = {str(k): str(v) for k, v in vars(training_args).items()}
            all_metrics["data_args"] = {str(k): str(v) for k, v in vars(data_args).items()}
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

    def load_last_checkpoint(self, training_args):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        print("last_checkpoint", last_checkpoint)
        best_model_path = os.path.join(last_checkpoint, "pytorch_model.bin")
        if os.path.exists(best_model_path):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(best_model_path, map_location="cpu")
            # If the model is on the GPU, it still works!

            load_result = self.model.load_state_dict(state_dict, strict=False)

            if len(load_result.missing_keys) != 0:
                if self.model._keys_to_ignore_on_save is not None \
                        and set(load_result.missing_keys) == set(self.model._keys_to_ignore_on_save):
                    self.model.tie_weights()
                else:
                    logger.warn(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
            if len(load_result.unexpected_keys) != 0:
                logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")


def my_rewrite_logs(d):
    new_d = defaultdict(dict)
    for k, v in d.items():
        if "train_" in k:
            new_d[k.replace("train_", "")]["train"] = v
        elif "eval_" in k:
            new_d[k.replace("eval_", "")]["eval"] = v
        elif "test_" in k:
            new_d[k.replace("test_", "")]["test"] = v
    return new_d


class MetricCombinedTensorBoardCallback(TensorBoardCallback):
    # a callback for tensorboard
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = my_rewrite_logs(logs)
            for k, v in logs.items():
                if type(v) is not dict:
                    self.tb_writer.add_scalars(k, v, state.epoch)  # state.epoch rather than global_step for tensorboard

            self.tb_writer.flush()
