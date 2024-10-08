# run like this "python data_scripts/download_datasets.py"
# in order to match the relative paths in the script
import os
import sys
import joblib
import pandas as pd
import requests

sys.path.append(os.getcwd())  # for relative imports

from data_scripts import read_qrels_file, read_docs_file, read_query_file, read_run_file


def download_qrcd2023_A():
    ########################
    # the dataset for QuranQA TASK-A 2023
    ########
    os.makedirs("biencoder/DRhard/data/QQA/dataset/", exist_ok=True)
    doc_file_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/Thematic_QPC/QQA23_TaskA_QPC_v1.1.tsv"
    train_query_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/QQA23_TaskA_ayatec_v1.2_train.tsv"
    dev_query_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/QQA23_TaskA_ayatec_v1.2_dev.tsv"
    test_query_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/QQA23_TaskA_ayatec_v1.2_test.tsv"
    dev_qrel_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/qrels/QQA23_TaskA_ayatec_v1.2_qrels_dev.gold"
    train_qrel_file_url = "https://gitlab.com/bigirqu/quran-qa-2023/-/raw/main/Task-A/data/qrels/QQA23_TaskA_ayatec_v1.2_qrels_train.gold"

    # Added URLs for downloading Tafseer files
    muyassar_file_url = "https://raw.githubusercontent.com/Q-Translate/translations/master/ar.muyassar.txt"
    jalalayn_file_url = "https://raw.githubusercontent.com/Q-Translate/translations/master/ar.jalalayn.txt"

    # Paths to save the files
    dev_query = "data/QQA23_TaskA_dev.tsv"
    doc_file = "data/QQA23_TaskA_QPC_v1.1.tsv"
    dev_qrel = "data/QQA23_TaskA_qrels_dev.gold"
    train_qrel = "data/QQA23_TaskA_qrels_train.gold"
    train_query = "data/QQA23_TaskA_train.tsv"
    test_query = "data/QQA23_TaskA_test.tsv"
    muyassar_file = "data/ar.muyassar.txt"
    jalalayn_file = "data/ar.jalalayn.txt"

    # Download files
    save_downloaded_file(train_query, train_query_file_url)
    save_downloaded_file(dev_query, dev_query_file_url)
    save_downloaded_file(doc_file, doc_file_file_url)
    save_downloaded_file(dev_qrel, dev_qrel_file_url)
    save_downloaded_file(train_qrel, train_qrel_file_url)
    save_downloaded_file(test_query, test_query_file_url)
    save_downloaded_file(muyassar_file, muyassar_file_url)
    save_downloaded_file(jalalayn_file, jalalayn_file_url)

    # saving files for STAR algorithm
    # 1) query files
    train_q_df = read_query_file(train_query_file_url)
    dev_q_df = read_query_file(dev_query_file_url)
    query_df = pd.concat([train_q_df, dev_q_df])

    query_df["new_qid"] = range(query_df.shape[0])
    qid_id_mapper = {row["qid"]: row["new_qid"] for _, row in query_df.iterrows()}
    train_q_df.loc[:, "qid"] = train_q_df["qid"].apply(lambda qid: qid_id_mapper[qid])
    dev_q_df.loc[:, "qid"] = dev_q_df["qid"].apply(lambda qid: qid_id_mapper[qid])
    train_q_df[["qid", "query_text"]].to_csv(f"biencoder/DRhard/data/QQA/dataset/train-query.tsv", sep="\t", index=False, header=False)
    dev_q_df[["qid", "query_text"]].to_csv(f"biencoder/DRhard/data/QQA/dataset/dev-query.tsv", sep="\t", index=False, header=False)

    # 2) doc file
    doc_df = read_docs_file(doc_file)
    doc_df["new_docid"] = range(doc_df.shape[0])
    doc_df[["new_docid", "doc_text"]].to_csv(f"biencoder/DRhard/data/QQA/dataset/docs.tsv", sep="\t", index=False, header=False)

    # 3) qrel files
    train_qrel_df = read_qrels_file(train_qrel)
    dev_qrel_df = read_qrels_file(dev_qrel)

    doc_id_mapper = {row["docid"]: row["new_docid"] for _, row in doc_df.iterrows()}

    train_qrel_df.loc[:, "docid"] = train_qrel_df["docid"].apply(lambda docid: doc_id_mapper.get(docid, "-1"))
    dev_qrel_df.loc[:, "docid"] = dev_qrel_df["docid"].apply(lambda docid: doc_id_mapper.get(docid, "-1"))

    train_qrel_df.loc[:, "qid"] = train_qrel_df["qid"].apply(lambda qid: qid_id_mapper.get(qid, "-1"))
    dev_qrel_df.loc[:, "qid"] = dev_qrel_df["qid"].apply(lambda qid: qid_id_mapper.get(qid, "-1"))

    train_qrel_pos_df = train_qrel_df[train_qrel_df["docid"] != "-1"]
    dev_qrel_pos_df = dev_qrel_df[dev_qrel_df["docid"] != "-1"]

    # with unanswerable
    train_qrel_df[["qid", "Q0", "docid", "relevance", ]].to_csv(f"biencoder/DRhard/data/QQA/dataset/train-qrel-full.tsv", sep="\t", index=False, header=False)
    dev_qrel_df[["qid", "Q0", "docid", "relevance", ]].to_csv(f"biencoder/DRhard/data/QQA/dataset/dev-qrel-full.tsv", sep="\t", index=False, header=False)

    # unanswerable removed
    train_qrel_pos_df[["qid", "Q0", "docid", "relevance", ]].to_csv(f"biencoder/DRhard/data/QQA/dataset/train-qrel-pos.tsv", sep="\t", index=False, header=False)
    dev_qrel_pos_df[["qid", "Q0", "docid", "relevance", ]].to_csv(f"biencoder/DRhard/data/QQA/dataset/dev-qrel-pos.tsv", sep="\t", index=False, header=False)
    joblib.dump(doc_id_mapper, "biencoder/DRhard/data/QQA/dataset/doc_id_mapper.dmp", compress=3)
    joblib.dump(qid_id_mapper, "biencoder/DRhard/data/QQA/dataset/qid_id_mapper.dmp", compress=3)
    print(train_query_file_url)


def save_downloaded_file(save_path, file_url):
    if not os.path.exists(save_path):
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
        else:
            print(f"Error {response.status_code}: Could not download file from {file_url}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_qrcd2023_A()
