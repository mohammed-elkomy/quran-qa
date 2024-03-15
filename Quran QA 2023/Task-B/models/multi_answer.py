import numpy as np
import torch

from torch.nn import CrossEntropyLoss


def multi_answer_loss(model, start_positions, end_positions, start_logits, end_logits, ignored_index):
    loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
    ################################################

    start_losses = [
        loss_fct(start_logits, start_positions_)
        for start_positions_ in torch.unbind(start_positions, dim=1)
    ]  # this loop does max_answers_in_batch iterations
    end_losses = [
        loss_fct(end_logits, end_positions_)
        for end_positions_ in torch.unbind(end_positions, dim=1)
    ]  # this loop does max_answers_in_batch iterations

    start_loss = torch.cat([loss.unsqueeze(1) for loss in start_losses], dim=1)
    end_loss = torch.cat([loss.unsqueeze(1) for loss in end_losses], dim=1)
    loss_tensor = (start_loss + end_loss) / 2
    ################################################

    if model.loss_type == "MAL":
        # the sum of all loss for all answers
        return loss_tensor.sum()  # reduce sum on answers and then reduce sum on samples
    elif model.loss_type == "first":
        # the first answer in every sample is only considered for loss computation and others don't contribute to the loss
        return loss_tensor[..., 0].sum()
    elif model.loss_type == "random":
        # a random answer in every sample is only considered for loss computation and others don't contribute to the loss
        rand_idx = np.random.randint(0, loss_tensor.shape[1])
        return loss_tensor[..., rand_idx].sum()
    else:
        raise NotImplementedError
