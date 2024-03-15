from collections import defaultdict
import re
import string

import pandas as pd


def is_tab_sparated(preditions_file_path):
    with open(preditions_file_path) as tsvfile:
        pair_ids = {}
        for line_no, line_str in enumerate(tsvfile, start=1):
            line = line_str.split("\t")
            if len(line) == 1:
                return False
    return True


def read_tafseer(file_path):
    with open(file_path, ) as f:
        data = f.readlines()
    tafser_data = [e.split("|") for e in data if len(e.split("|")) == 3]

    tafseer_dict = defaultdict(dict)
    for surah, aya, tafser_text in tafser_data:
        tafseer_dict[surah][aya] = tafser_text
    return tafseer_dict


def read_qrels_file(qrels_file):
    # split_token = "\t" if format_checker.is_tab_sparated(qrels_file) else  "\s+"
    df_qrels = pd.read_csv(qrels_file, sep="\t", names=["qid", "Q0", "docid", "relevance"])
    df_qrels["qid"] = df_qrels["qid"].astype(str)
    df_qrels["docid"] = df_qrels["docid"].astype(str)
    return df_qrels


def read_docs_file(docs_file):
    doc_df = pd.read_csv(docs_file, sep="\t", names=["docid", "doc_text"])
    doc_df["docid"] = doc_df["docid"].astype(str)
    return doc_df


def read_query_file(query_file):
    query_df = pd.read_csv(query_file, sep="\t", names=["qid", "query_text"])
    query_df["qid"] = query_df["qid"].astype(str)
    return query_df


def read_run_file(run_file):
    # since the run is definitely is space or tab separated
    # identify the separator token based on the file separator token
    split_token = "\t" if is_tab_sparated(run_file) else "\s+"
    df_run = pd.read_csv(run_file, sep=split_token, names=["qid", "Q0", "docid", "rank", "score", "tag"])
    df_run["qid"] = df_run["qid"].astype(str)
    df_run["docid"] = df_run["docid"].astype(str)
    return df_run



arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def clean_text(text):
    text = remove_punctuations(text)
    text = remove_diacritics(text)

    return text
