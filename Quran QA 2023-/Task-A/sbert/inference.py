import pandas as pd
from sentence_transformers import util


def infer_relevance(encoder, query_dataset, doc_dataset, tok_k_relevant=1000):
    def encode_normalize(passages):
        embeddings = encoder.encode(passages, convert_to_tensor=True).to('cuda')
        normalized_embeddings = util.normalize_embeddings(embeddings)
        return normalized_embeddings

    query_dataset = query_dataset[["query_text", "qid"]].drop_duplicates()
    query_dataset.reset_index(drop=True, inplace=True)
    doc_dataset.reset_index(drop=True, inplace=True)
    query_embeddings = encode_normalize(query_dataset["query_text"])
    collection = encode_normalize(doc_dataset["doc_text"])

    queries_hits = util.semantic_search(query_embeddings, collection, score_function=util.cos_sim, top_k=tok_k_relevant)

    result = {"qid": [], "Q0": [], "docid": [], "rank": [], "score": []}
    for query_idx, query_hits in enumerate(queries_hits):
        qid = query_dataset["qid"][query_idx]
        for rank, hit in enumerate(query_hits):
            docid = doc_dataset["docid"][hit['corpus_id']]
            score = hit['score']
            result["qid"].append(qid)
            result["score"].append(score)
            result["docid"].append(docid)
            result["rank"].append(rank)
            result["Q0"].append("Q0")
    result = pd.DataFrame(result)
    result['docid'] = result['docid'].astype(str)
    return result[["qid", "Q0", "docid", "rank", "score", ]]
