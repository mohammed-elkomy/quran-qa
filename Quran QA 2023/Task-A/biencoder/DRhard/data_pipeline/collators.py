
import os
import sys
import torch

sys.path.append(os.getcwd())  # for relative imports

from biencoder.DRhard.data_pipeline import pack_tensor_2D


def get_adore_eval_collate_function(max_seq_length):
    cnt = 0

    def collate_function(batch):
        """

        @param batch:  a list of
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
        }
        @return: collated batch
        """
        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids

    return collate_function


def get_adore_train_collate_function(max_seq_length):
    cnt = 0

    def collate_function(batch):
        """

                @param batch:  a list of
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,  # equal importance
                    "id": item,
                    'rel_ids': [pid1, pid2, pid3]
                     "neg_ids": [npid1, npid2, npid3,]
                }
                @return: collated batch
        """

        nonlocal cnt
        length = None
        if cnt < 10:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
        qids = [x['id'] for x in batch]
        all_rel_pids = [x["rel_ids"] for x in batch]
        all_irrel_pids = [x.get("neg_ids", []) for x in batch]
        return data, all_rel_pids, all_irrel_pids

    return collate_function


def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0

    def collate_function(batch):
        """

         @param batch:  a list of
         {
             "input_ids": input_ids,
             "attention_mask": attention_mask,  # equal importance
             "id": item,
         }
         @return: collated batch
        """
        nonlocal cnt
        length = None
        if cnt < 10 or padding:
            length = max_seq_length
            cnt += 1

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
        ids = [x['id'] for x in batch]
        return data, ids

    return collate_function


#### I think this function is not referenced any where
# def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
#     query_collate_func = single_get_collate_function(max_query_length, padding)
#     doc_collate_func = single_get_collate_function(max_doc_length, padding)
#
#     def collate_function(batch):
#         query_data, query_ids = query_collate_func([x[0] for x in batch])
#         doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
#         rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0
#                           for docid in doc_ids]
#                          for qid in query_ids]
#         input_data = {
#             "input_query_ids": query_data['input_ids'],
#             "query_attention_mask": query_data['attention_mask'],
#             "input_doc_ids": doc_data['input_ids'],
#             "doc_attention_mask": doc_data['attention_mask'],
#             "rel_pair_mask": torch.FloatTensor(rel_pair_mask),
#         }
#         return input_data
#
#     return collate_function


def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False, hard_neg_dataset=True):
    ## STAR COLLATOR, this will be fed to rand_inbatch_neg_train or hardneg_train forward functions

    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)
    """ 
    the output single_get_collate_function format: 
    data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
    ids = [qid_or_pid]
    """

    def collate_function(batch):
        """
        for each entry query_data, passage_data, hard_passage_data
        the format is
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
        }

        @param batch: a list of triples [(query_data, passage_data, hard_passage_data)] each of which has the format above
        @return:
        """
        query_data, query_ids = query_collate_func([x[0] for x in batch])  # passing a list of dicts to single_get_collate_function
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])  # passing a list of dicts to single_get_collate_function
        # query_ids are not token ids
        # doc_ids are not token ids

        # sum([[1], [2], [30]], [55]) >===>  [55, 1, 2, 30] flatten
        neg_doc_data, neg_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        # we will get a list of hard negatives (they are relevant but the model failed to predict them) : TrainInbatchWithHardDataset
        # neg_doc_ids may also come from a random in-batch negative sampler : TrainInbatchWithRandDataset

        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0
                          for docid in doc_ids]
                         for qid in query_ids]  # (0) just relevant or (1) irrelevant(random inbatch negatives)    # BATCH X BATCH

        neg_or_hard_pair_mask = [[1 if docid not in rel_dict[qid] else 0
                                  for docid in neg_doc_ids]
                                 for qid in query_ids]  # (0) just relevant  or (1) irrelevant(random inbatch negatives + hard inbatch negative)   # BATCH X [BATCH x HARD_NEG]

        query_num = len(query_data['input_ids'])  # number of queries
        hard_num_per_query = len(batch[0][2])  # it's equal for all queries
        input_data = {
            "input_query_ids": query_data['input_ids'],  # BATCH x SEQ_LEN
            "query_attention_mask": query_data['attention_mask'],  # BATCH x SEQ_LEN (value = 1)
            "input_doc_ids": doc_data['input_ids'],  # BATCH x SEQ_LEN
            "doc_attention_mask": doc_data['attention_mask'],  # BATCH x SEQ_LEN (value = 1)
            # i verified this reshaping is contagious as numpy
            "other_doc_ids": neg_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),  # convert to 3d mat  BATCH x NEG x SEQ_LEN
            "other_doc_attention_mask": neg_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),  # convert to 3d mat , BATCH x NEG x SEQ_LEN  (value = 1)

        }
        if hard_neg_dataset:
            input_data["rel_pair_mask"] = torch.FloatTensor(rel_pair_mask)  # just relevant or  irrelevant(inbatch negatives)
            input_data["hard_pair_mask"] = torch.FloatTensor(neg_or_hard_pair_mask)  # (relevant and hard) or irrelevant(in batch negatives)
        else:
            input_data["rand_neg_pair_mask"] = torch.FloatTensor(neg_or_hard_pair_mask)  # (relevant and hard) or irrelevant(in batch negatives)

        return input_data

    return collate_function


def pair_get_collate_function(max_query_length, max_doc_length, padding=False):
    ## STAR COLLATOR for labelled only dataset (no hard or random negative involved),
    # will be fed to labelled_only_train forward function

    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)
    """ 
    the output single_get_collate_function format: 
    data = {
            "input_ids": pack_tensor_2D(input_ids, default=1,
                                        dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_2D(attention_mask, default=0,
                                             dtype=torch.int64, length=length),
        }
    ids = [qid_or_pid]
    """

    def collate_function(batch):
        """
        for each entry query_data, passage_data, label
        the format is
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
        }

        @param batch: a list of triples [(query_data, passage_data, label)] each of which has the format above, label is a float 1(relevant) or -1(irrelevant)
        @return:
        """
        query_data, query_ids = query_collate_func([x[0] for x in batch])  # passing a list of dicts to single_get_collate_function
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])  # passing a list of dicts to single_get_collate_function
        # query_ids are not token ids
        # doc_ids are not token ids

        labels = torch.FloatTensor([x[2] for x in batch])

        input_data = {
            "input_query_ids": query_data['input_ids'],  # BATCH x SEQ_LEN
            "query_attention_mask": query_data['attention_mask'],  # BATCH x SEQ_LEN (value = 1)
            "input_doc_ids": doc_data['input_ids'],  # BATCH x SEQ_LEN
            "doc_attention_mask": doc_data['attention_mask'],  # BATCH x SEQ_LEN (value = 1)
            "labels": labels
        }
        return input_data

    return collate_function
