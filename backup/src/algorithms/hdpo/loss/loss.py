from src import torch, nn
from src.algorithms.base import BaseAlgorithm
from src.envs.inventory.simulator import Simulator

class PolicyLoss(nn.Module):
    """
    Loss that returns the sum of the rewards.
    """
    
    def __init__(self):
        super(PolicyLoss, self).__init__()

    def forward(self, observation, action, reward):
        return reward.sum()