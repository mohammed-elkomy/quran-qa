import joblib
import pandas as pd
from joypy import joyplot
from matplotlib import pyplot as plt


def get_aggregated_df(df, ):
    concat_agg = lambda x: ','.join([str(x) for x in x.unique()])
    aggregate_command = dict.fromkeys(df.columns, "mean")
    for k in ['model_name', 'split_name', 'seed', 'lr', 'loss_type', 'subfolder', 'qa_pretrained', 'dump_file', ]:
        aggregate_command[k] = concat_agg

    summarized = df.groupby("subfolder").agg(aggregate_command)

    summarized["faithful"] = summarized["subfolder"].str.contains("faithful")
    summarized["group"] = summarized.index

    return summarized


def get_summary_df(source):
    source_df = pd.DataFrame(source)
    rm_cols = [c for c in source_df.columns if "per_sample" in c]
    return source_df.drop(columns=rm_cols)


def make_ridgeline(data, columns, by, path, title, ):
    fig, axes = joyplot(
        data=data,
        by=by,
        column=columns,
        legend=True,
        alpha=0.6,
        figsize=(18, 10),
        range_style="own",
        tails=0,  # exact x-range
        title=title,
        xlabelsize=15,
        ylabelsize=15,
        grid="y"
        # hist=True,  overlap=0,
        # grid=True,
    )
    fig.savefig(path, dpi=150)
    plt.close()


def ensemble_to_summary(ensemble_results):
    ensemble_results_df = []
    for model_dir, model_results in ensemble_results.items():
        filter_dict = {}
        for kk, entry_results in model_results.items():
            if kk in ["raw_eval_ensemble", "post_eval_ensemble", "optimal_ensemble_pAP@10","num_models"]:
                if type(entry_results) == dict and "per_sample" in entry_results:
                    entry_results.pop("per_sample")
                filter_dict[kk] = entry_results
        filter_dict["model_dir"] = model_dir.replace("dumps/original", "")
        ensemble_results_df.append(pd.json_normalize(filter_dict))

    return pd.concat(ensemble_results_df)


ensemble_results_df = ensemble_to_summary(joblib.load("artifacts/summary/ensemble-eval.dmp"))
faithful = joblib.load("artifacts/summary/faithful.dump")
original = joblib.load("artifacts/summary/original.dump")
original_df = get_summary_df(original)
faithful_df = get_summary_df(faithful)

original_agg = get_aggregated_df(original_df)
faithful_agg = get_aggregated_df(faithful_df)
original_agg.merge(ensemble_results_df, right_on="model_dir", left_on="group").to_excel("artifacts/summary/original.xlsx")
faithful_agg.to_excel("artifacts/summary/faithful.xlsx")


original_df["draw"] = original_df["subfolder"].str.replace("artifacts//", "").str.replace("/", " ➡ ")
faithful_df["draw"] = faithful_df["subfolder"].str.replace("artifacts//", "").str.replace("/", " ➡ ")

for col in ['eval_official_pAP@10', 'eval_best_pAP_thresh', 'eval_pAP@10', 'eval_best_pAP',
            'post_eval_official_pAP@10', 'post_eval_best_pAP_thresh', 'post_eval_pAP@10', 'post_eval_best_pAP', ]:
    make_ridgeline(data=original_df,
                   columns=[col],
                   path=f"/Core/Workspaces/jetBrains/pycharm/QuranQA2023-B/artifacts/kde/orig/{col}.png",
                   title=f"Ridgeline {col} study",
                   by="draw", )

    make_ridgeline(data=faithful_df,
                   columns=[col],
                   path=f"/Core/Workspaces/jetBrains/pycharm/QuranQA2023-B/artifacts/kde/faith/{col}.png",
                   title=f"Ridgeline {col} study",
                   by="draw", )