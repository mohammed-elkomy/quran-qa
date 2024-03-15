

import transformers
from torch import nn

from model import BaseModelDot
from model.forward_functions import hardneg_train, rand_inbatch_neg_train, labelled_only_train

if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from transformers import RobertaModel


class RobertaDot(BaseModelDot, RobertaPreTrainedModel):
    def __init__(self, config, model_argobj=None):
        # config.model_type
        BaseModelDot.__init__(self, model_argobj)
        RobertaPreTrainedModel.__init__(self, config)
        if int(transformers.__version__[0]) == 4:
            config.return_dict = False
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            self.output_embedding_size = config.hidden_size
        print("output_embedding_size", self.output_embedding_size)
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.norm = nn.LayerNorm(self.output_embedding_size)
        self.apply(self._init_weights)  # recursive initialization

    def _text_encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        return outputs1


class RobertaDot_Hard(RobertaDot):
    """hard negatives + inbatch negatives"""
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return hardneg_train(self.query_emb, self.body_emb,
                             input_query_ids, query_attention_mask,
                             input_doc_ids, doc_attention_mask,
                             other_doc_ids, other_doc_attention_mask,
                             rel_pair_mask, hard_pair_mask)


class RobertaDot_RandInBatch(RobertaDot):
    """random inbatch negatives"""
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                other_doc_ids=None, other_doc_attention_mask=None,
                rel_pair_mask=None, hard_pair_mask=None):
        return rand_inbatch_neg_train(self.query_emb, self.body_emb,
                                      input_query_ids, query_attention_mask,
                                      input_doc_ids, doc_attention_mask,
                                      other_doc_ids, other_doc_attention_mask,
                                      hard_pair_mask)  # maybe it's better if we get the input in rel_pair_mask

class RobertaDot_Labelled(RobertaDot):
    """random inbatch negatives"""
    def forward(self, input_query_ids, query_attention_mask,
                input_doc_ids, doc_attention_mask,
                labels):
        return labelled_only_train(self.query_emb, self.body_emb,
                                      input_query_ids, query_attention_mask,
                                      input_doc_ids, doc_attention_mask,
                                      labels
                                     )
