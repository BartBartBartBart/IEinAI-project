import torch
from collections import OrderedDict
from typing import Tuple

import torch

def trajectory_balance_loss(total_flow, rewards, fwd_probs):
    """
    Computes the mean trajectory balance loss for a collection of samples. For
    more information, see Bengio et al. (2022): https://arxiv.org/abs/2201.13259
    
    Args:
        total_flow: The estimated total flow used by the GFlowNet when drawing
        the collection of samples for which the loss should be computed
        
        rewards: The rewards associated with the final state of each of the
        samples
        
        fwd_probs: The forward probabilities associated with the trajectory of
        each sample (i.e. the probabilities of the actions actually taken in
        each trajectory)
        
        back_probs: The backward probabilities associated with each trajectory
    """
    # for some reason, the fwd_probs might contain zero, we then discard any
    # sample with such fwd_prob
    lhs = total_flow * torch.prod(fwd_probs, dim=1)
    # rhs = rewards * torch.prod(back_probs, dim=1)
    
    # Check for no zero values in lhs and rewards
    assert (lhs != 0).all(), "Total flow or forward probabilities contain zero values"
    assert (rewards != 0).all(), "Rewards contain zero values"
    # check for finite values in lhs and rewards
    assert torch.isfinite(lhs).all(), "Total flow or forward probabilities contain non-finite values"
    # print("REWARDS", rewards)

    # assert torch.isfinite(rewards).all(), "Rewards contain non-finite values"
    # Replace inf values in rewards with 1e-8 to avoid numerical issues
    rewards[~torch.isfinite(rewards)] = 1e-8

    loss = torch.log(lhs / rewards + 1e-8)**2
    # print("LOSS", loss)
    assert torch.isfinite(loss).all(), total_flow
    # if not torch.isfinite(loss).all():
    #     s = (torch.prod(fwd_probs, dim=1) == 0).sum()
    #     print(s)
    #     print(loss[lhs != 0].mean())
    return loss.mean()


class LossBuffer(object):
    """
    A data structure that stores the reward values for equations already seen.
    An equation is considered as "seen"/"visited" before up to all constant values. A special
    `update_interval` parameter is required to control how often this buffer "forgets" the
    rewards for update purpose (the reward function can be dynamic).

    Args:
        buffer_size: the maximum capacity of distinct expression trees in the buffer.
        update_interval: the number of times after which the buffer will forget an expression.
    """
    def __init__(self, buffer_size=5000, update_interval=1000):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.cache = OrderedDict()
        self.counter = dict()

    def get(self, encoding: torch.Tensor):
        encoding = tuple(encoding.tolist())
        if encoding not in self.cache:
            return None
        if encoding not in self.counter:
            self.cache.move_to_end(encoding)
            return self.cache[encoding]
        else:
            remain_count = self.counter[encoding]
            if remain_count == 0:
                return None
            else:
                self.cache.move_to_end(encoding)
                self.counter[encoding] = remain_count - 1
                return self.cache[encoding]

    def set(self, encoding: torch.Tensor, has_constant: bool, loss: torch.Tensor):
        encoding = tuple(encoding.tolist())
        if has_constant:
            self.counter[encoding] = self.update_interval
            curr_loss = self.cache.get(encoding, None)
            if curr_loss is not None:
                self.cache[encoding] = torch.min(loss, curr_loss)
            else:
                self.cache[encoding] = loss
        else:
            self.cache[encoding] = loss

        self.cache.move_to_end(encoding)
        if len(self.cache) > self.buffer_size:
            k, _ = self.cache.popitem(last=False)
            self.counter.pop(k, None)








