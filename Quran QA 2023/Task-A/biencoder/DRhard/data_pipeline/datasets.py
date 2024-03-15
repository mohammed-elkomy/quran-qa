
import os
import sys
import json
import random
from torch.utils.data import Dataset

sys.path.append(os.getcwd())  # for relative imports

from biencoder.DRhard.data_pipeline import SequenceDataset, load_rel


class TrainInbatchDataset(Dataset):
    """
    to train a dense-retrieval system using only positives
    """

    def __init__(self, rel_file, query_ids_cache, doc_ids_cache,
                 max_query_length, max_doc_length):
        self.query_dataset = SequenceDataset(query_ids_cache, max_query_length)
        self.doc_dataset = SequenceDataset(doc_ids_cache, max_doc_length)
        self.reldict, _ = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    # def __getitem__(self, item):
    #     """
    #     this method is never called i think
    #     @param item: random id
    #     @return: a positive pair of query and document
    #     """
    #     qid = self.qids[item]
    #     pid = random.choice(self.reldict[qid])  # positive document
    #
    #     """
    #     format
    #     {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,  # equal importance
    #         "id": item,
    #     }
    #     """
    #     query_data = self.query_dataset[qid]  # get the token ids that where saved during the preprocessing phase
    #     passage_data = self.doc_dataset[pid]  # get the token ids that where saved during the preprocessing phase
    #     return query_data, passage_data


class TrainInbatchWithHardDataset(TrainInbatchDataset):
    """
    train using hard negatives
    each sample is a query, pos_doc, [neg_docs]
    """

    def __init__(self, rel_file, hard_json_file, query_ids_cache,
                 doc_ids_cache, hard_num,
                 max_query_length, max_doc_length):
        """
        @param rel_file: label file, trec format
        @param hard_json_file: predictions json file, generated using prepare_hardneg.py
        @param query_ids_cache: cache of token ids for queries
        @param doc_ids_cache: cache of token ids for documents
        @param hard_num: number of hard negatives to mine
        @param max_query_length:
        @param max_doc_length:
        """
        TrainInbatchDataset.__init__(self,
                                     rel_file, query_ids_cache, doc_ids_cache,
                                     max_query_length, max_doc_length)
        self.hard_dict = json.load(open(hard_json_file))
        assert hard_num > 0
        self.hard_num = hard_num

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        """
        each sample is a query, pos_doc, [neg_docs]
        """
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])  # random positive document

        """
        format of query_data, passage_data
        {
        "input_ids": input_ids,
        "attention_mask": attention_mask,  # equal importance
        "id": item,
        }
        """
        query_data = self.query_dataset[qid]  # get the token ids that where saved during the preprocessing phase
        passage_data = self.doc_dataset[pid]  # get the token ids that where saved during the preprocessing phase
        # str(qid) because rankdict was jsonified
        hard_pids = random.sample(self.hard_dict[str(qid)], self.hard_num)
        hard_passage_data = [self.doc_dataset[hard_pid] for hard_pid in hard_pids]  # a list of hard negatives that were not in rel_dict

        # this will be collated with triple_get_collate_function
        return query_data, passage_data, hard_passage_data


class TrainInbatchWithRandDataset(TrainInbatchDataset):
    """
    same as hard negatives but using random negatives, we don't need "rankdict" file
    """

    def __init__(self, rel_file, query_ids_cache,
                 doc_ids_cache, rand_num,
                 max_query_length, max_doc_length):
        TrainInbatchDataset.__init__(self,
                                     rel_file, query_ids_cache, doc_ids_cache,
                                     max_query_length, max_doc_length)
        assert rand_num > 0
        self.rand_num = rand_num

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])  # random positive document

        """
          format of query_data, passage_data
          {
          "input_ids": input_ids,
          "attention_mask": attention_mask,  # equal importance
          "id": item,
          }
          """
        query_data = self.query_dataset[qid]  # get the token ids that where saved during the preprocessing phase
        passage_data = self.doc_dataset[pid]  # get the token ids that where saved during the preprocessing phase
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]  # a list of random ids

        # this will be collated with triple_get_collate_function
        return query_data, passage_data, rand_passage_data


class TrainWithLabelledOnlyDataset(TrainInbatchDataset):
    """
    trains a model without mining hard or random negatives, just the annotated pairs by humans
    """

    def __init__(self, rel_file, query_ids_cache,
                 doc_ids_cache,
                 max_query_length, max_doc_length):
        """
        @param rel_file: label file, trec format
        @param hard_json_file: predictions json file, generated using prepare_hardneg.py
        @param query_ids_cache: cache of token ids for queries
        @param doc_ids_cache: cache of token ids for documents
        @param max_query_length:
        @param max_doc_length:
        """
        TrainInbatchDataset.__init__(self,
                                     rel_file, query_ids_cache, doc_ids_cache,
                                     max_query_length, max_doc_length)

        self.reldict, self.neg_dict = load_rel(rel_file)
        assert self.neg_dict  # assuming the qrel file has negative judgements
        self.qrel = []
        for qid, pids in self.reldict.items():
            for pid in pids:
                self.qrel.append((qid, pid, 1.0))

        for qid, n_pids in self.neg_dict.items():
            for n_pid in n_pids:
                self.qrel.append((qid, n_pid, -1.0))

    def __len__(self):
        return len(self.qrel)

    def __getitem__(self, item):
        """
        each sample is a random query, doc_doc, relevant(1)/irrelevant(-1)
        """
        qid, pid, label = self.qrel[item]

        # if random.random() > .5:
        #     label = 1.0
        #     pid = random.choice(self.reldict[qid])  # random positive document
        # else:
        #     label = -1.0
        #     pid = random.choice(self.neg_dict[qid])  # random negative document
        """
        format of query_data, passage_data
        {
        "input_ids": input_ids,
        "attention_mask": attention_mask,  # equal importance
        "id": item,
        }
        """
        query_data = self.query_dataset[qid]  # get the token ids that where saved during the preprocessing phase
        passage_data = self.doc_dataset[pid]  # get the token ids that where saved during the preprocessing phase

        # this will be collated with pair_get_collate_function
        return query_data, passage_data, label


class ADORETrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        """
        a SequenceDataset returns this
         {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
         }  
         # this will be collated by collate function
        """
        self.rel_dict, self.neg_dict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_ids'] = self.rel_dict[item]  # add rel_id to the dict
        if self.neg_dict:
            # add neg_id to the dict, those are labelled as negatives in the dataset, not hard negatives mined by adore
            ret_val['neg_ids'] = self.neg_dict[item]

        """
        ret_val now has 
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # equal importance
            "id": item,
            "rel_ids": [pid1, pid2, pid3]
            "neg_ids": [npid1, npid2, npid3,]
         }  
        """
        return ret_val
