import glob
import json
import os
from collections import defaultdict

import numpy as np
from tabulate import tabulate

#################################
# how to use this script?
# just run it with no cmd args, and it will print tables for all trained checkpoints for the eval phase
#################################
ROOT_PATH = "results/eval"
#################################
HEADERS = ["Model name", "eval_exact_match", "eval_f1", "eval_pRR", "eval_optimal_pRR@5", "seed", "epoch"]

table_data = []
for dump_file in glob.glob(os.path.join(ROOT_PATH, "**/*.dump")):
    model_name = os.path.split(dump_file)[-1].replace(".dump", "").rsplit("-", 1)
    try:
        with open(dump_file.replace("dump", "json").replace("eval-", "").replace("predict-", ""), 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        assert metrics["seed"] == int(model_name[1])
        table_data.append([
            model_name[0],
        ])
        table_data[-1].extend([metrics[header] for header in HEADERS[1:]])
        try:
            best_model_step = int(metrics["best_model"].rsplit("-", 1)[-1])
            num_epochs = metrics["epoch"]
            best_model_epoch = best_model_step / metrics["total_steps"] * num_epochs
            table_data[-1].append(best_model_epoch)
        except:
            table_data[-1].append("---")
    except:
        pass

table_data.sort(key=lambda item: (item[0], item[3]), reverse=True)

model_tables = defaultdict(list)
for row in table_data:
    model_name = row[0]
    model_tables[model_name].append(row)

HEADERS.append("best_model_at_epoch")
HEADERS = ["Index"] + HEADERS
for model_table in model_tables.values():
    _, eval_exact_match, eval_f1, eval_pRR, eval_optimal_pRR_5, _, epochs, best_model_at_epoch = list(zip(*model_table))
    eval_exact_match = np.mean(eval_exact_match)
    eval_f1 = np.mean(eval_f1)
    eval_pRR = np.mean(eval_pRR)
    eval_optimal_pRR_5 = np.mean(eval_optimal_pRR_5)
    try:
        best_model_at_epoch = np.mean(best_model_at_epoch)
    except:
        best_model_at_epoch = "---"
    for idx, model_row in enumerate(model_table, start=1):
        model_row.insert(0, idx)
    model_table.append(["--", "AVERAGE", eval_exact_match, eval_f1, eval_pRR, eval_optimal_pRR_5, "---", "---", best_model_at_epoch])

    table = tabulate(model_table, headers=HEADERS, tablefmt="fancy_grid", stralign="center", numalign="center")
    table = table.replace("   AVERAGE    ", "★★AVERAGE★★")
    print(table, end="\n\n\n")
