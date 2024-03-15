import os
import sys

from transformers import BertForQuestionAnswering

from transformers.modeling_outputs import QuestionAnsweringModelOutput

sys.path.append(os.getcwd())  # for relative imports

from models.multi_answer import multi_answer_loss


class MultiAnswerBertForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config, loss_type):
        super().__init__(config)
        self.loss_type = loss_type  # sum of multi-answers losses

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        """
        @param input_ids: same as BertForQuestionAnswering
        @param attention_mask: same as BertForQuestionAnswering
        @param token_type_ids: same as BertForQuestionAnswering
        @param position_ids: same as BertForQuestionAnswering
        @param head_mask: same as BertForQuestionAnswering
        @param inputs_embeds: same as BertForQuestionAnswering
        @param output_attentions: same as BertForQuestionAnswering
        @param output_hidden_states: same as BertForQuestionAnswering
        @param return_dict: same as BertForQuestionAnswering

        # for multi-answer support, we have different losses for this maximum marginal likelihood / hard learning, more on that here https://arxiv.org/abs/1909.04849
        @param start_positions: Labels for position (index) [batch_size, max_answers_in_batch], the original from BertForQuestionAnswering is [batch_size,]
        @param end_positions: Labels for position (index) [batch_size, max_answers_in_batch], the original from BertForQuestionAnswering is [batch_size,]
        @return: same as BertForQuestionAnswering
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 2:  # [batch x max_answers x MULTI_GPU_DIM]
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 2:  # [batch x max_answers x MULTI_GPU_DIM]
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            total_loss = multi_answer_loss(self, start_positions, end_positions, start_logits, end_logits, ignored_index,)

        # print(f"\n\n\nloss {total_loss}\n\n",)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
