import sys

sys.path += ['./']
import torch
from torch import nn

import torch.nn.functional as F
from torch.cuda.amp import autocast


def hardneg_train(query_encode_func, doc_encode_func,
                  input_query_ids, query_attention_mask,
                  input_doc_ids, doc_attention_mask,
                  other_doc_ids=None, other_doc_attention_mask=None,
                  rel_pair_mask=None, hard_pair_mask=None):
    """
    a forward function to be called by a model to be trained using 1) in-batch negatives 2) static hard negatives
    @param query_encode_func: the embedding function for a query
    @param doc_encode_func: the embedding function for a document
    @param input_query_ids: token ids for a query
    @param query_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param input_doc_ids: token ids for a document
    @param doc_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param other_doc_ids: token ids for a negatives documents (includes hard + random negatives)
    @param other_doc_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param rel_pair_mask: a mask for relevance
    @param hard_pair_mask: a mask for hard-negative irrelevant pairs # (0) just relevant  or (1) irrelevant(random inbatch negatives + hard inbatch negative)
    @return: scalar loss for an optimizer
    """
    # queries and documents are of the same length,
    # other docs give N negatives for each query
    query_embs = query_encode_func(input_query_ids, query_attention_mask)  # BATCH X FEATURES
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)  # BATCH X FEATURES

    # random negatives
    batch_size = query_embs.shape[0]
    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)  # BATCH X BATCH
        # print("batch_scores", batch_scores)
        single_positive_scores = torch.diagonal(batch_scores, 0)  # only diagonal elements, positive paris  [pid_1_qid_1, pid_2_qid_2,...]
        # print("positive_scores", positive_scores)
        # repeat the positives to compute the pairwise loss
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, batch_size).reshape(-1)  # repeat [pid_1_qid_1, pid_1_qid_1, pid_1_qid_1, pid_1_qid_1,.. pid_2_qid_2,...]
        if rel_pair_mask is None:
            rel_pair_mask = 1 - torch.eye(batch_size, dtype=batch_scores.dtype, device=batch_scores.device)  # zero diagonal,  # BATCH X BATCH
            # print("mask", mask)
        batch_scores = batch_scores.reshape(-1)

        logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                  batch_scores.unsqueeze(1)], dim=1)  # (BATCH X BATCH) x 2, this is the pairwise loss
        # print(logit_matrix)
        # computing the ranknet loss, you can derive it on a piece of paper in 2 steps :D, hint: -1 to swap numer / denum
        lsm = F.log_softmax(logit_matrix, dim=1)  # (BATCH X BATCH) x 2, softmax to normalize, log to get the log loss
        # lsm is always negative, softmax means the values are less than 1, log(value<1) is negative
        # relevant pairs have no loss (mask zero)
        # the first is the positive score, the second is the negative score
        # if we select the positive score (i.e 0) after the softmax we have a quantifiable relevance loss
        # for ex if we consider the most offending example, [0,+ve] => softmax => [0,1]  => log => [-inf, 0] => +inf loss !
        loss = -1.0 * lsm[:, 0] * rel_pair_mask.reshape(-1)  # (BATCH X BATCH) x 1

        # print(loss)
        # print("\n")
        first_loss, first_num = loss.sum(), rel_pair_mask.sum()

    if other_doc_ids is None:
        return (first_loss / first_num,)

    # other_doc_ids[3D]: BATCH x NEG x SEQ_LEN
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)  # [BATCH x NEG] x SEQ_LEN
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)  # [BATCH x NEG] x SEQ_LEN
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)  # [BATCH x NEG] X FEATURES

    # hard negatives
    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)  # BATCH X [BATCH x NEG]
        other_batch_scores = other_batch_scores.reshape(-1)  # [BATCH X BATCH x NEG]
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)  # repeat single_positive_scores as long as the negatives [BATCH X BATCH x NEG]
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                        other_batch_scores.unsqueeze(1)], dim=1)  # [BATCH X BATCH x NEG] x 2, this is the pairwise loss with hard-negatives
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)  # this is always negative
        other_loss = -1.0 * other_lsm[:, 0]  # [BATCH X BATCH x NEG] x 1
        # print(loss)
        # print("\n")
        if hard_pair_mask is not None:
            hard_pair_mask = hard_pair_mask.reshape(-1)  # [BATCH X BATCH x NEG] x 1
            other_loss = other_loss * hard_pair_mask
            second_loss, second_num = other_loss.sum(), hard_pair_mask.sum()
        else:
            # no mask, all are ones, it's rare to find relevant pairs in hard-negatives we mine
            second_loss, second_num = other_loss.sum(), len(other_loss)

    return ((first_loss + second_loss) / (first_num + second_num),)


def rand_inbatch_neg_train(query_encode_func, doc_encode_func,
                           input_query_ids, query_attention_mask,
                           input_doc_ids, doc_attention_mask,
                           other_doc_ids=None, other_doc_attention_mask=None,
                           rand_neg_pair_mask=None):
    """
    a forward function to be called by a model for in-batch negatives training scheme
    @param query_encode_func: the embedding function for a query
    @param doc_encode_func: the embedding function for a document
    @param input_query_ids: token ids for a query
    @param query_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param input_doc_ids: token ids for a document
    @param doc_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param other_doc_ids: token ids for a negatives documents (includes hard + random negatives)
    @param other_doc_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param rand_neg_pair_mask: a mask for relevance, it has rand_num for every query sampled randomly
    @return: scalar loss for an optimizer
    """
    query_embs = query_encode_func(input_query_ids, query_attention_mask)
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)

    with autocast(enabled=False):
        batch_scores = torch.matmul(query_embs, doc_embs.T)  # [BATCH X BATCH]
        single_positive_scores = torch.diagonal(batch_scores, 0)  # select diagonals , [BATCH]

    # other_doc_ids[3D]: BATCH x NEG x SEQ_LEN
    other_doc_num = other_doc_ids.shape[0] * other_doc_ids.shape[1]  # [BATCH x NEG]
    other_doc_ids = other_doc_ids.reshape(other_doc_num, -1)  # [BATCH x NEG] x SEQ_LEN
    other_doc_attention_mask = other_doc_attention_mask.reshape(other_doc_num, -1)  # [BATCH x NEG] x SEQ_LEN
    other_doc_embs = doc_encode_func(other_doc_ids, other_doc_attention_mask)  # [BATCH x NEG] X FEATURES

    with autocast(enabled=False):
        other_batch_scores = torch.matmul(query_embs, other_doc_embs.T)  # BATCH x [BATCH x NEG]
        other_batch_scores = other_batch_scores.reshape(-1)  # [BATCH X BATCH x NEG]
        positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, other_doc_num).reshape(-1)  # repeat single_positive_scores as long as the negatives [BATCH X BATCH x NEG]
        other_logit_matrix = torch.cat([positive_scores.unsqueeze(1),
                                        other_batch_scores.unsqueeze(1)], dim=1)  # [BATCH X BATCH x NEG] x 2, this is the pairwise loss with hard-negatives
        # print(logit_matrix)
        other_lsm = F.log_softmax(other_logit_matrix, dim=1)  # [BATCH X BATCH x NEG] x 2
        other_loss = -1.0 * other_lsm[:, 0]  # [BATCH X BATCH x NEG] x 1
        if rand_neg_pair_mask is not None:
            rand_neg_pair_mask = rand_neg_pair_mask.reshape(-1)  # [BATCH X BATCH x NEG] x 1
            other_loss = other_loss * rand_neg_pair_mask
            second_loss, second_num = other_loss.sum(), rand_neg_pair_mask.sum()
        else:
            # no mask, all are ones
            second_loss, second_num = other_loss.sum(), len(other_loss)
    return (second_loss / second_num,)


def labelled_only_train(query_encode_func, doc_encode_func,
                        input_query_ids, query_attention_mask,
                        input_doc_ids, doc_attention_mask,
                        labels
                        ):
    """
    a forward function to be called by a model for labelled only dataset (no hard or random negative involved)
    here we have a list of pids, list of docs, a list of labels (they are equal), unlike rand_inbatch_neg_train and hardneg_train who build the pair-wise embedding_loss
    @param labels:
    @param query_encode_func: the embedding function for a query
    @param doc_encode_func: the embedding function for a document
    @param input_query_ids: token ids for a query
    @param query_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @param input_doc_ids: token ids for a document
    @param doc_attention_mask: attention mask for transformer-encoder, (1 = equal importance)
    @return: scalar embedding_loss for an optimizer
    """

    query_embs = query_encode_func(input_query_ids, query_attention_mask)  # [BATCH]
    doc_embs = doc_encode_func(input_doc_ids, doc_attention_mask)  # [BATCH]

    with autocast(enabled=False):
        # embedding_loss = nn.CosineEmbeddingLoss(margin=.25)(query_embs, doc_embs, labels)
        output = torch.cosine_similarity(query_embs, doc_embs)

        labels = (labels > 0).type(torch.FloatTensor).to(device=output.device)

        embedding_loss = nn.MSELoss()(output, labels.view(-1))
        return (embedding_loss,)  # scalar
