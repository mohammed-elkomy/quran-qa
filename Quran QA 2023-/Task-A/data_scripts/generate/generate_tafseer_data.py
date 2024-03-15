import os
import sys
from functools import partial
import pandas as pd

sys.path.append(os.getcwd())  # for relative imports
from data_scripts import read_tafseer, read_docs_file, clean_text

muyassar = read_tafseer("data/ar.muyassar.txt")
jalalayn = read_tafseer("data/ar.jalalayn.txt")


def expand_passage(doc_no, external):
    surah, aya = doc_no.split(":")
    s, e = aya.split("-")

    return clean_text(" ".join([external[surah][str(idx)] for idx in range(int(s), int(e) + 1)]))


full_data = read_docs_file("data/QQA23_TaskA_QPC_v1.1.tsv")

full_data["muyassar"] = full_data["docid"].apply(partial(expand_passage, external=muyassar), )
full_data["jalalayn"] = full_data["docid"].apply(partial(expand_passage, external=jalalayn), )
full_data["muyassar_q"] = [f"m-{idx:04d}" for idx in range(full_data.shape[0])]
full_data["jalalayn_q"] = [f"j-{idx:04d}" for idx in range(full_data.shape[0])]

full_data[["docid", "doc_text"]].to_csv(f"data/tafseer_docs.tsv", sep="\t", index=False, header=False)

query_data = pd.concat([full_data[["muyassar_q", "muyassar"]].rename(columns={"muyassar_q": "qid", "muyassar": "query_text"}),
                        full_data[["jalalayn_q", "jalalayn"]].rename(columns={"jalalayn_q": "qid", "jalalayn": "query_text"}), ])

qrel_df = pd.concat([full_data[["muyassar_q", "docid"]].rename(columns={"muyassar_q": "qid", }),
                     full_data[["jalalayn_q", "docid"]].rename(columns={"jalalayn_q": "qid", }), ])

qrel_df["Q0"] = "Q0"
qrel_df["relevance"] = "1"
qrel_df[["qid", "Q0", "docid", "relevance", ]].to_csv(f"data/tafseer-qrel.tsv", sep="\t", index=False, header=False)
query_data[["qid", "query_text"]].to_csv(f"data/tafseer-query.tsv", sep="\t", index=False, header=False)
