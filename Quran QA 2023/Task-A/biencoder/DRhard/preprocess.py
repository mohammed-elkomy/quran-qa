import argparse
import json
import multiprocessing
import os
import pickle
import subprocess
import sys

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.getcwd())  # for relative imports


# if int(transformers.__version__.split(".")[0]) >= 3:
#     raise RuntimeError("Please use Transformers Libarary version 2 for preprossing because \
#         we find RobertaTokenzier behaves differently between version 2 and 3/4")
#

def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    """
    pads an input to max_length
    """
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids


def tokenize_to_file(args, in_path, output_dir, line_fn, max_length, begin_idx, end_idx):
    """
    working on a slice from begin_idx to end_idx
    write 3 files
    "ids.memmap": from auto increment id to non standard qid_or_pid
    "token_ids.memmap": token ids from tokenizer
    "lengths.memmap": length per passage or query
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True, cache_dir=None)
    os.makedirs(output_dir, exist_ok=True)
    data_cnt = end_idx - begin_idx
    ids_array = np.memmap(
        os.path.join(output_dir, "ids.memmap"),
        shape=(data_cnt,), mode='w+', dtype=np.int32)

    token_ids_array = np.memmap(
        os.path.join(output_dir, "token_ids.memmap"),
        shape=(data_cnt, max_length), mode='w+', dtype=np.int32)

    token_length_array = np.memmap(
        os.path.join(output_dir, "lengths.memmap"),
        shape=(data_cnt,), mode='w+', dtype=np.int32)

    pbar = tqdm(total=end_idx - begin_idx, desc=f"Tokenizing")
    for idx, line in enumerate(open(in_path, 'r')):
        if idx < begin_idx:
            continue
        if idx >= end_idx:
            break
        qid_or_pid, token_ids, length = line_fn(args, line, tokenizer)
        write_idx = idx - begin_idx
        ids_array[write_idx] = qid_or_pid
        token_ids_array[write_idx, :] = token_ids
        token_length_array[write_idx] = length
        pbar.update(1)
    pbar.close()
    assert write_idx == data_cnt - 1


def multi_file_process(args, num_process, in_path, out_path, line_fn, max_length):
    """
    parallel calls to tokenize_to_file on different slices and save the splits to folders
    """
    output_linecnt = subprocess.check_output(["wc", "-l", in_path]).decode("utf-8")
    print("line cnt", output_linecnt)
    all_linecnt = int(output_linecnt.split()[0])
    run_arguments = []
    for i in range(num_process):
        begin_idx = round(all_linecnt * i / num_process)
        end_idx = round(all_linecnt * (i + 1) / num_process)
        output_dir = f"{out_path}_split_{i}"
        run_arguments.append((
            args, in_path, output_dir, line_fn,
            max_length, begin_idx, end_idx
        ))
    pool = multiprocessing.Pool(processes=num_process)
    pool.starmap(tokenize_to_file, run_arguments)
    pool.close()
    pool.join()
    splits_dir = [a[2] for a in run_arguments]
    return splits_dir, all_linecnt


def write_query_rel(args, pid2offset, qid2offset_file, query_file, input_qrel_file, out_query_file, standard_qrel_file):
    # sourcery skip
    """
    we should only tokenize the queries found in the label set qrel file
    we will process the queries in splits as we did in passages
    then we will merge those splits such that we don't copy queries not referenced by qrel
    @param args: cmd args
    @param pid2offset: mapping passage id to our autoincrement offset
    @param qid2offset_file: output file
    @param query_file: source query text to be tokenized
    @param input_qrel_file: input qrel file has the format         Q_ID	0	P_ID	RELEVANCE, RELEVANCE is not used for msmarco (when all qrel are positive)
    @param out_query_file: output tokenized queries
    @param standard_qrel_file: standardized qrel to output
    """

    print("Writing query files " + str(out_query_file) +
          " and " + str(standard_qrel_file))
    query_collection_path = os.path.join(args.data_dir, query_file)
    if input_qrel_file is None:
        query_positive_id = None
        valid_query_num = int(subprocess.check_output(["wc", "-l", query_collection_path]).decode("utf-8").split()[0])
    else:
        query_positive_id = set()
        input_qrel_file_path = os.path.join(
            args.data_dir,
            input_qrel_file,
        )

        print("Loading query_2_pos_docid")
        for line in open(input_qrel_file_path, 'r', encoding='utf8'):
            qid = line.split()[0]
            if qid[0] == "E":
                qid = qid[1:]
            assert line.split()[-1] == "1"
            query_positive_id.add(int(qid))
        valid_query_num = len(query_positive_id)
        # valid_query_num is the queries ids in the label set
        if args.data_type == "QQA":
            # QQA dataset has orphan queries we need to count them too
            valid_query_num = int(subprocess.check_output(["wc", "-l", query_collection_path]).decode("utf-8").split()[0])

    out_query_path = os.path.join(args.out_data_dir, out_query_file, )

    qid2offset = {}

    print('start query file split processing')
    splits_dir_lst, _ = multi_file_process(
        args, args.threads, query_collection_path,
        out_query_path, QueryPreprocessingFn,
        args.max_query_length
    )

    print('start merging splits')

    token_ids_array = np.memmap(
        out_query_path + ".memmap",
        shape=(valid_query_num, args.max_query_length), mode='w+', dtype=np.int32)

    token_length_array = []

    idx = 0
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)

        split_token_ids_array = np.memmap(os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)

        split_token_length_array = np.memmap(os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)

        for q_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            if query_positive_id is not None and q_id not in query_positive_id and args.data_type != "QQA":
                # exclude the query as it is not in label set
                # dont exclude any queries from QQA (we have queries that are negative against all docs)
                continue

            # here we will only add the queries found in the label set(in qrel)
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length)
            qid2offset[q_id] = idx
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(q_id))

    # idx : number of queries ids in the label set
    # token_ids_array.shape[0], valid_query_num is the queries ids in the label set
    # token_length_array this will hold the length for each query in the label set
    assert len(token_length_array) == len(token_ids_array) == idx
    print("MAX query length", max(token_length_array))
    np.save(out_query_path + "_length.npy", np.array(token_length_array))

    qid2offset_path = os.path.join(args.out_data_dir, qid2offset_file, )

    with open(qid2offset_path, 'wb') as handle:
        pickle.dump(qid2offset, handle, protocol=4)
    print("done saving qid2offset")

    print("Total lines written: " + str(idx))
    meta = {'type': 'int32', 'total_number': idx,
            'embedding_size': args.max_query_length}  # misnamed ?
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)

    if input_qrel_file is None:
        print("No qrels file provided")
        return

    print("Writing qrels")
    with open(os.path.join(args.out_data_dir, standard_qrel_file), "w", encoding='utf-8') as standard_qrel_output:
        out_line_count = 0
        for line in open(input_qrel_file_path, 'r', encoding='utf8'):
            topicid, _, docid, rel = line.split()

            if args.data_type == 'doc':  # doc
                docid = int(docid[1:])  # remove D
            elif args.data_type == 'passage':  # passage
                docid = int(docid)
            elif args.data_type == 'QQA':  # Quran QA task-a
                docid = int(docid)

            topicid = int(topicid)

            standard_qrel_output.write(str(qid2offset[topicid]) +
                                       "\t0\t" + str(pid2offset[docid]) +
                                       "\t" + rel + "\n")  # standardize ids from dataset id to offset
            out_line_count += 1
        print("Total lines written: " + str(out_line_count))


def preprocess(args):
    pid2offset = {}

    if args.data_type == 'doc':  # doc
        in_passage_path = os.path.join(
            args.data_dir,
            "msmarco-docs.tsv",
        )
    elif args.data_type == 'passage':  # passage
        in_passage_path = os.path.join(
            args.data_dir,
            "collection.tsv",
        )
    elif args.data_type == 'QQA':  # Quran QA task-a
        in_passage_path = os.path.join(
            args.data_dir,
            "docs.tsv",
        )
    else:
        raise Exception("unknown datatype")

    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    print('start passage file split processing')
    splits_dir_lst, all_linecnt = multi_file_process(
        args, args.threads, in_passage_path,
        out_passage_path, PassagePreprocessingFn,
        args.max_seq_length
    )

    token_ids_array = np.memmap(
        out_passage_path + ".memmap",
        shape=(all_linecnt, args.max_seq_length), mode='w+', dtype=np.int32)
    token_length_array = []

    idx = 0
    out_line_count = 0
    print('start merging splits')
    for split_dir in splits_dir_lst:
        ids_array = np.memmap(os.path.join(split_dir, "ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = np.memmap(os.path.join(split_dir, "token_ids.memmap"), mode='r', dtype=np.int32)
        split_token_ids_array = split_token_ids_array.reshape(len(ids_array), -1)
        split_token_length_array = np.memmap(os.path.join(split_dir, "lengths.memmap"), mode='r', dtype=np.int32)

        for p_id, token_ids, length in zip(ids_array, split_token_ids_array, split_token_length_array):
            token_ids_array[idx, :] = token_ids
            token_length_array.append(length)
            pid2offset[p_id] = idx  # maps the id in the passages tsv file to an auto-increment id
            idx += 1
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1
    assert len(token_length_array) == len(token_ids_array) == idx
    print("MAX passage length", max(token_length_array))
    np.save(out_passage_path + "_length.npy", np.array(token_length_array))

    print("Total lines written: " + str(out_line_count))

    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}

    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)

    pid2offset_path = os.path.join(args.out_data_dir, "pid2offset.pickle", )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)

    print("done saving pid2offset")

    if args.data_type == 'doc':  # doc
        """
        # train
        1) msmarco-doctrain-queries.tsv
        1) msmarco-doctrain-qrels.tsv ############### exactly one pid per qid

        # dev
        2) msmarco-docdev-queries.tsv
        2) msmarco-docdev-qrels.tsv ############### exactly one pid per qid

        # TREC
        3) msmarco-test2019-queries.tsv
        3) 2019qrels-docs.txt ############### has relevance scores other than 1 
        
        # MARCO leaderboard
        4) docleaderboard-queries.tsv
        """
        write_query_rel(
            args,  ########################### IN: input params
            pid2offset,  ##################### IN: standard pid mapping
            "train-qid2offset.pickle",  ###### OUT: standard qid mapping
            "msmarco-doctrain-queries.tsv",  # IN: quires text to be tokenized
            "msmarco-doctrain-qrels.tsv",  ### IN: non standard qrel : non standard pid >  non standard qid, used to find all qids in ,
            "train-query",  ################## OUT: the tokenized data
            "train-qrel.tsv")  ############### OUT: the standard qrel (remapping ids)  standard pid >  standard qid

        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-docs.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "msmarco-docdev-queries.tsv",
            "msmarco-docdev-qrels.tsv",
            "dev-query",
            "dev-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "docleaderboard-queries.tsv",
            None,  # no input qrel
            "lead-query",
            None)  # no output qrel
    elif args.data_type == 'passage':  # passage
        """
        # train
        1) queries.train.tsv
        1) qrels.train.tsv ############### at least 1 pids per qid, at most 7 pids per qid
        # dev
        2) queries.dev.small.tsv
        2) qrels.dev.small.tsv ############### at least 1 pids per qid, at most 4 pids per qid
        # 
        3) msmarco-test2019-queries.tsv 
        3) 2019qrels-pass.txt ############### has relevance scores other than 1 
        # eval
        4) queries.eval.small.tsv
        """

        write_query_rel(
            args,  ###################### IN: input params
            pid2offset,  ################ IN: standard pid mapping
            "train-qid2offset.pickle",  # OUT: standard qid mapping
            "queries.train.tsv",  ####### IN: quires text to be tokenized
            "qrels.train.tsv",  ######### IN: non standard qrel : non standard pid >  non standard qid, used to find all qids in
            "train-query",  ############# OUT: the tokenized data
            "train-qrel.tsv")  ########## OUT: the standard qrel (remapping ids)  standard pid >  standard qid

        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "queries.dev.small.tsv",
            "qrels.dev.small.tsv",
            "dev-query",
            "dev-qrel.tsv")

        write_query_rel(
            args,
            pid2offset,
            "test-qid2offset.pickle",
            "msmarco-test2019-queries.tsv",
            "2019qrels-pass.txt",
            "test-query",
            "test-qrel.tsv")
        write_query_rel(
            args,
            pid2offset,
            "lead-qid2offset.pickle",
            "queries.eval.small.tsv",
            None,
            "lead-query",
            None)
    elif args.data_type == 'QQA':  # Quran QA task-a

        """
        # train
        1) train-query.tsv
        1) train-qrel-pos.tsv
        # dev
        2) dev-query.tsv
        2) dev-qrel-pos.tsv 
        """

        write_query_rel(
            args,  ###################### IN: input params
            pid2offset,  ################ IN: standard pid mapping
            "train-qid2offset.pickle",  # OUT: standard qid mapping
            "train-query.tsv",  ####### IN: quires text to be tokenized
            "train-qrel-pos.tsv",  ######### IN: non standard qrel : non standard pid >  non standard qid
            "train-query",  ############# OUT: the tokenized data
            "train-qrel.tsv")  ########## OUT: the standard qrel (remapping ids)  standard pid >  standard qid

        # leaking dev to training (testing the correctness of the code)
        # write_query_rel(
        #     args,  ###################### IN: input params
        #     pid2offset,  ################ IN: standard pid mapping
        #     "train-qid2offset.pickle",  # OUT: standard qid mapping
        #     "dev-query.tsv",  ####### IN: quires text to be tokenized
        #     "dev-qrel-pos.tsv",  ######### IN: non standard qrel : non standard pid >  non standard qid
        #     "train-query",  ############# OUT: the tokenized data
        #     "train-qrel.tsv")  ########## OUT: the standard qrel (remapping ids)  standard pid >  standard qid

        write_query_rel(
            args,
            pid2offset,
            "dev-qid2offset.pickle",
            "dev-query.tsv",
            "dev-qrel-pos.tsv",
            "dev-query",
            "dev-qrel.tsv")


def PassagePreprocessingFn(args, line, tokenizer):
    """
    @param args: cmd args
    @param line: passage or document line from tsv format
    @param tokenizer: hugging face tokenizer
    @return: extracted id, padded tokens, sequence length
    """
    if args.data_type == 'doc':  # doc
        # D14561, URL, TITLE, TEXT
        line_arr = line.split('\t')
        p_id = int(line_arr[0][1:])  # remove "D"

        url = line_arr[1].rstrip()
        title = line_arr[2].rstrip()
        p_text = line_arr[3].rstrip()
        # NOTE: This linke is copied from ANCE, 
        # but I think it's better to use <s> as the separator, 
        full_text = url + "<sep>" + title + "<sep>" + p_text
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = full_text[:args.max_doc_character]
    elif args.data_type == 'passage':  # passage
        # 14561, TEXT

        line = line.strip()
        line_arr = line.split('\t')
        p_id = int(line_arr[0])

        p_text = line_arr[1].rstrip()
        # keep only first 10000 characters, should be sufficient for any
        # experiment that uses less than 500 - 1k tokens
        full_text = p_text[:args.max_doc_character]
    elif args.data_type == 'QQA':  # Quran QA task-a
        line_arr = line.split('\t')
        p_id = int(line_arr[0])
        p_text = line_arr[1].strip()
        full_text = p_text[:args.max_doc_character]
    else:
        raise Exception("unknown datatype")

    passage = tokenizer.encode(
        full_text,
        add_special_tokens=True,  # to handle <sep>
        max_length=args.max_seq_length,
        truncation=True
    )  # return the passage ids only
    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id, input_id_b, passage_len


def QueryPreprocessingFn(args, line, tokenizer):
    """
    @param args: cmd args
    @param line: query line from tsv format
    @param tokenizer: hugging face tokenizer
    @return: extracted id, padded tokens, sequence length
    """
    line_arr = line.split('\t')
    q_id = line_arr[0]
    q_id = int(q_id)

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        max_length=args.max_query_length,
        truncation=True)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id, input_id_b, passage_len


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,  # since this code will support other transformers than roberta, I will feed it manually roberta-base
        type=str,
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )
    # parser.add_argument(
    #     "--data_type",
    #     default=1,
    #     type=int,
    #     help="0 for doc, 1 for passage",
    # )
    parser.add_argument("--data_type", choices=["passage", 'doc', "QQA"], type=str, required=True)
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()

    return args


def main():
    args = get_arguments()
    if args.data_type == 'doc':  # doc
        args.data_dir = "./data/doc/dataset"
        args.out_data_dir = "./data/doc/preprocess"
    elif args.data_type == 'passage':  # passage
        args.data_dir = "./data/passage/dataset"
        args.out_data_dir = "./data/passage/preprocess"
    elif args.data_type == 'QQA':  # QQA
        args.data_dir = "./data/QQA/dataset"
        args.out_data_dir = "./data/QQA/preprocess"
    else:
        raise Exception("unknown datatype")

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)


if __name__ == '__main__':
    main()
