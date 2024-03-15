import os
import pickle
import argparse

import joblib
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["passage", "doc", "QQA"], required=True)
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, f"{args.mode}.rank.tsv")
    output_path = os.path.join(args.output_dir, f"{args.mode}.rank.tsv")
    # assert not os.path.exists(output_path)
    os.makedirs(args.output_dir, exist_ok=True)

    pid2offset = pickle.load(open(os.path.join(args.preprocess_dir, "pid2offset.pickle"), 'rb'))
    offset2pid = {v: k for k, v in pid2offset.items()}
    qid2offset = pickle.load(open(os.path.join(args.preprocess_dir, f"{args.mode}-qid2offset.pickle"), 'rb'))
    offset2qid = {v: k for k, v in qid2offset.items()}

    if args.dataset == "QQA":
        qid_id_mapper = joblib.load(os.path.join(args.preprocess_dir, "..", "dataset", "qid_id_mapper.dmp"))  # old to new
        doc_id_mapper = joblib.load(os.path.join(args.preprocess_dir, "..", "dataset", "doc_id_mapper.dmp"))  # old to new
        qid_id_mapper = {v: k for k, v in qid_id_mapper.items()}  # new to old
        doc_id_mapper = {v: k for k, v in doc_id_mapper.items()}  # new to old

    with open(output_path, 'w') as output:
        for line in tqdm(open(input_path)):
            qid, pid, rank, score = line.split()
            qid, pid, rank = int(qid), int(pid), int(rank)
            qid, pid = offset2qid[qid], offset2pid[pid]
            if args.dataset == "doc":
                output.write(f"{qid}\tD{pid}\t{rank}\n")
            elif args.dataset == "passage":
                output.write(f"{qid}\t{pid}\t{rank}\n")
            elif args.dataset == "QQA":
                qid = qid_id_mapper[qid]
                pid = doc_id_mapper[pid]
                output.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score}\n")
            else:
                raise "unknown dataset"

    print(output_path, "saved")

"""
--input_dir
./data/QQA/evaluate/star/
--preprocess_dir
./data/QQA/preprocess
--output_dir
./data/QQA/official_runs/star
--mode
dev
--dataset
QQA
"""
