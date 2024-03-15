import pandas as pd


def infer_relevance(model, inference_df, tok_k_relevant=1000):
    inference_examples = [[str(line["query_text"]), str(line["doc_text"]), ] for _, line in inference_df.iterrows()]
    predicted_scores = model.predict(inference_examples, show_progress_bar=False)  # apply_softmax=True, only for multi classification

    inference_df["score"] = predicted_scores

    dfs = []
    for q_id, q_df in inference_df.groupby("qid"):
        q_df = q_df.sort_values("score", ascending=False).head(tok_k_relevant)
        q_df["rank"] = list(range(q_df.shape[0]))
        dfs.append(q_df)
    results_df = pd.concat(dfs)
    results_df = results_df[["qid", "Q0", "docid", "rank", "score", "tag"]]
    return results_df
