import os.path
import re
import shutil
from functools import partial

import pandas as pd
from snowballstemmer import stemmer
import arabicstopwords.arabicstopwords as ar_stp
import pyterrier as pt
import subprocess
from collections import defaultdict

from data_scripts import read_tafseer

TOP_K = 1000

# pyterrier is a Python API for Terrier. Link: https://github.com/terrier-org/pyterrier
# Terrier IR Platform is a modular open source software for the rapid development of large-scale information retrieval applications.
if not pt.started():
    pt.init(helper_version="0.0.6", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

# arabic stemmer
ar_stemmer = stemmer("arabic")

# define some global constants
TEXT = "text"
QUERY = "query"
LABEL = "label"
RANK = "rank"
TAG = "tag"
SCORE = "score"
QID = "qid"
DOC_NO = "docno"
DOCID = "docid"


# Clean text from urls, handles, special characters, tabs, line jumps, and extra white space.
def clean(text):
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text)  # remove special characters
    text = re.sub(r"\t", " ", text)  # remove tabs
    text = re.sub(r"\n", " ", text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space
    text = re.sub(r"[^\w\s]", "", text)  # Removing punctuations in string using regex
    text = text.strip()
    return text


# remove arabic stop words
def ar_remove_stop_words(sentence):
    terms = []
    stopWords = set(ar_stp.stopwords_list())
    for term in sentence.split():
        if term not in stopWords:
            terms.append(term)
    return " ".join(terms)


# normalize the arabic text
def normalize_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return (text)


# stem the arabic text
def ar_stem(sentence):
    return " ".join([ar_stemmer.stemWord(i) for i in sentence.split()])


# apply all preprocessing steps needed for Arabic text
def preprocess_arabic(text):
    text = normalize_arabic(text)
    text = ar_remove_stop_words(text)
    text = ar_stem(text)
    return text


def prepare_query_for_search(query_path, query_column=TEXT,
                             id_column=DOC_NO):
    names = [DOC_NO, TEXT]
    print("Cleaning queries and applying preprocessing steps")
    df_query = read_file(query_path, names=names)
    # apply the cleaning functions on the queries/questions
    df_query[QUERY] = df_query[query_column].apply(clean)

    # apply normalization, stemming and stop word removal
    print("Applying normalization, stemming and stop word removal")
    df_query[QUERY] = df_query[QUERY].apply(preprocess_arabic)

    df_query[QID] = df_query[id_column].astype(str)  # convert the id column to string
    df_query = df_query[[QID, QUERY]]  # keep the columns needed for search
    print("Done with preparation!")
    return df_query


def load_index(index_path):
    try:
        index = pt.IndexFactory.of(index_path)
        print("Index was loaded successfully from this path: ", index_path)
        return index
    except Exception as e:
        print("Cannot load the index, check exception details {}".format(e))
        return []


# read file based on its extension (tsv or xlsx)
def read_file(input_file, sep="\t", names=""):
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    else:
        if names != "":
            df = pd.read_csv(input_file, sep=sep, names=names, encoding="utf-8")
        else:
            df = pd.read_csv(input_file, sep=sep, encoding="utf-8")
    return df


def make_inference(IR_model, query_file, run_file_name):
    # 2. read the query file and prepare it for search to match pyterrier format
    df_query = prepare_query_for_search(query_file)
    # 3. search using BM25 model
    df_run = IR_model.transform(df_query)
    # 4. save the run in trec format to a file
    df_run["Q0"] = ["Q0"] * len(df_run)
    df_run["tag"] = ["BM25"] * len(df_run)
    df_run[["qid", "Q0", "docno", "rank", "score", "tag"]].to_csv(f"artifacts/{run_file_name}", sep="\t", index=False, header=False)


def print_eval(run_file_name, gold_file_name):
    # Set the command to call the script with the arguments
    command = ["python", "metrics/QQA23_TaskA_eval.py",
               "-r", os.path.join("artifacts", run_file_name), "-q", os.path.join("data", gold_file_name)]
    # Call the command using subprocess
    subprocess.run(command)


def expand_passage(entry, external):
    surah, aya = entry["docno"].split(":")
    s, e = aya.split("-")
    entry["text"] = entry["text"] + " ".join([external[surah][str(idx)] for idx in range(int(s), int(e) + 1)])
    return entry


def expand_passages(full_data, external):
    full_data = full_data.apply(partial(expand_passage, external=external), axis=1)
    return full_data


if __name__ == "__main__":
    #############################
    muyassar = read_tafseer("data/ar.muyassar.txt")
    jalalayn = read_tafseer("data/ar.jalalayn.txt")

    full_data = read_file("data/QQA23_TaskA_QPC_v1.1.tsv", names=["docno", "text"])
    indexer = pt.DFIndexer(os.path.join(os.getcwd(), "artifacts/plain"), overwrite=True, tokeniser="UTFTokeniser")
    # full_data = expand_passages(full_data, muyassar)
    full_data["text"] = full_data["text"].apply(clean)
    full_data["text"] = full_data["text"].apply(preprocess_arabic)

    index_ref = indexer.index(full_data["text"], full_data["docno"])
    index_ref.toString()

    # we will first load the index
    index = pt.IndexFactory.of(index_ref)
    # we will call getCollectionStatistics() to check the stats
    print(index.getCollectionStatistics().toString())
    #############################

    # 1. initialize the BM25 retrieval model
    BM25_model = pt.BatchRetrieve(index, controls={"wmodel": "BM25"}, num_results=TOP_K)

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    BM25_model_bo1 = BM25_model >> bo1 >> BM25_model

    make_inference(BM25_model, "data/QQA23_TaskA_train.tsv", "TCE_BM25train.tsv")
    make_inference(BM25_model, "data/QQA23_TaskA_dev.tsv", "TCE_BM25dev.tsv")

    make_inference(BM25_model_bo1, "data/QQA23_TaskA_train.tsv", "TCE_QeBM25T.tsv")
    make_inference(BM25_model_bo1, "data/QQA23_TaskA_dev.tsv", "TCE_QeBM25D.tsv")

    print("\nBM25, train")
    print_eval(run_file_name="TCE_BM25train.tsv", gold_file_name="QQA23_TaskA_qrels_train.gold")

    print("\nBM25, dev")
    print_eval(run_file_name="TCE_BM25dev.tsv", gold_file_name="QQA23_TaskA_qrels_dev.gold")

    print("\nBM25 with bo1, train")
    print_eval(run_file_name="TCE_QeBM25T.tsv", gold_file_name="QQA23_TaskA_qrels_train.gold")

    print("\nBM25 with bo1, dev")
    print_eval(run_file_name="TCE_QeBM25D.tsv", gold_file_name="QQA23_TaskA_qrels_dev.gold")
    exit()
    ##################################################
    # STAR DATA
    # full_data = read_file("/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/docs.tsv", names=["docno", "text"])
    # indexer = pt.DFIndexer(os.path.join(os.getcwd(), "artifacts/plain"), overwrite=True, tokeniser="UTFTokeniser")
    # # full_data = expand_passages(full_data, muyassar)
    # full_data["text"] = full_data["text"].apply(clean)
    # full_data["text"] = full_data["text"].apply(preprocess_arabic)
    # full_data["docno"] = full_data["docno"].astype(str)
    # index_ref = indexer.index(full_data["text"], full_data["docno"])
    # index_ref.toString()
    #
    # # we will first load the index
    # index = pt.IndexFactory.of(index_ref)
    # # we will call getCollectionStatistics() to check the stats
    # print(index.getCollectionStatistics().toString())
    # #############################
    # #############################
    #
    # # 1. initialize the BM25 retrieval model
    # BM25_model = pt.BatchRetrieve(index, controls={"wmodel": "BM25"}, num_results=TOP_K)
    #
    # bo1 = pt.rewrite.Bo1QueryExpansion(index)
    # BM25_model_bo1 = BM25_model >> bo1 >> BM25_model
    #
    # make_inference(BM25_model, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/train-query.tsv", "TCE_BM25train.tsv")
    # make_inference(BM25_model, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/dev-query.tsv", "TCE_BM25dev.tsv")
    #
    # make_inference(BM25_model_bo1, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/train-query.tsv", "TCE_QeBM25T.tsv")
    # make_inference(BM25_model_bo1, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/dev-query.tsv", "TCE_QeBM25D.tsv")
    #
    # print("\nBM25, train")
    # print_eval(run_file_name="TCE_BM25train.tsv", gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/train-qrel-full.tsv")
    #
    # print("\nBM25, dev")
    # print_eval(run_file_name="TCE_BM25dev.tsv", gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/dev-qrel-full.tsv")
    #
    # print("\nBM25 with bo1, train")
    # print_eval(run_file_name="TCE_QeBM25T.tsv", gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/train-qrel-full.tsv")
    #
    # print("\nBM25 with bo1, dev")
    # print_eval(run_file_name="TCE_QeBM25D.tsv", gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/biencoder/DRhard/data/QQA/dataset/dev-qrel-full.tsv")

    full_data = read_file("/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/tafseer_docs.tsv", names=["docno", "text"])
    indexer = pt.DFIndexer(os.path.join(os.getcwd(), "artifacts/plain"), overwrite=True, tokeniser="UTFTokeniser")
    # full_data = expand_passages(full_data, muyassar)
    full_data["text"] = full_data["text"].apply(clean)
    full_data["text"] = full_data["text"].apply(preprocess_arabic)
    full_data["docno"] = full_data["docno"].astype(str)
    index_ref = indexer.index(full_data["text"], full_data["docno"])
    index_ref.toString()

    # we will first load the index
    index = pt.IndexFactory.of(index_ref)
    # we will call getCollectionStatistics() to check the stats
    print(index.getCollectionStatistics().toString())
    #############################
    #############################

    # 1. initialize the BM25 retrieval model
    BM25_model = pt.BatchRetrieve(index, controls={"wmodel": "BM25"}, num_results=TOP_K)

    bo1 = pt.rewrite.Bo1QueryExpansion(index)
    BM25_model_bo1 = BM25_model >> bo1 >> BM25_model

    make_inference(BM25_model, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/tafseer-query.tsv", "TAF_BM25train.tsv")
    make_inference(BM25_model_bo1, "/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/tafseer-query.tsv", "TAF_QeBM25T.tsv")

    print("\nBM25, train")
    print_eval(run_file_name="TAF_BM25train.tsv",
               gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/tafseer-qrel.tsv")

    print("\nBM25 with bo1, train")
    print_eval(run_file_name="TAF_QeBM25T.tsv",
               gold_file_name="/Core/Workspaces/jetBrains/pycharm/QuranQA2023-A/data/tafseer-qrel.tsv")
