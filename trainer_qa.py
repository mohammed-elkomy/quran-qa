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

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self,
                 eval_dataset=None,
                 eval_examples=None,
                 ignore_keys=None,
                 metric_key_prefix: str = "eval"):
        print("RUNNING EVALUATION STEP", "\n\n")
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop  # use_legacy_prediction_loop defaults on false
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)

            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
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

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        return self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict"
        )

    def save_metrics(self, split, metrics, seed, combined=True):
        if not self.is_world_process_zero():
            return
        print("my save metrics")
        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = f"{os.path.split(self.args.output_dir)[-1]}-{seed}.json"

            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            all_metrics["seed"] = seed
            all_metrics["best_model"] = self.state.best_model_checkpoint
            all_metrics["total_steps"] = self.state.global_step

            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)


