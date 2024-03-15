import glob
from zipfile import ZipFile

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from analysis.perfromance_analysis import read_from_zip
from analysis.retrieval_ensemble import retrieval_ensemble
from data_scripts import read_run_file, read_qrels_file
from metrics.Custom_TaskA_eval import evaluation_after_thresh, evaluate_task_a, convert_to_dict

plt.style.use(['science'])  # ,'no-latex'
plt.rcParams.update({
    "font.size": 12,
    "font.weight": 13,
})


def bold(text):
    return f"\\begin{{center}} \\textbf{{{text}}} \\end{{center}}"


run_file = "artifacts/dev_perfect.tsv"
df_run = read_run_file(run_file)

model_runs = [read_from_zip(ZipFile(f, "r"),
                            "eval_inference.tsv") for f in glob.glob("artifacts/dumps/bert-base-arabic-camelbert-ca-tydi-tafseer-pairs/**")]

# df_run = retrieval_ensemble(model_runs)

for idx, df_run in enumerate(model_runs):
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(9)

    qrels_file = "data/QQA23_TaskA_qrels_dev.gold"
    df_qrels = read_qrels_file(qrels_file)

    print(evaluate_task_a(df_qrels, df_run, ))
    query_to_doc_preds = convert_to_dict(df_run, column1="qid", column2="docid", column3="score")
    preds = {k: len(v) > 0 for k, v in query_to_doc_preds.items()}  # whether each query is predicted to have and answer or not
    na_probs = {k: -sum(v.values()) for k, v in query_to_doc_preds.items()}  # negatively proportional to the total doc scores
    min_lim, max_lim = min(na_probs.values()), max(na_probs.values())
    na_probs = {k: (v - min_lim) / (max_lim - min_lim) for k, v in na_probs.items()}

    x_vals = []
    y_vals = []
    for thresh in np.arange(0, 1.001, .05):
        x = evaluation_after_thresh(df_qrels, df_run, thresh)
        x_vals.append(thresh)
        y_vals.append(x[0]["overall"]["map_cut_10"])
    plt.plot(x_vals, y_vals)
    ax.set_yticklabels([f"\\textbf{{{ytick:0.2f}}}" for ytick in ax.get_yticks()])
    ax.set_xticklabels([f"\\textbf{{{xtick:0.2f}}}" for xtick in ax.get_xticks()])

    plt.xlabel(bold('Threshold'))
    plt.ylabel(bold('MAP performance (\%)', ))
    # plt.title(bold('Effect of thresholding', ))
    # Displaying the chart

    plt.savefig(f"latex/00_figs/{idx}.pdf", format="pdf")
    plt.close()
