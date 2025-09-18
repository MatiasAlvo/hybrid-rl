from src import torch, nn
from src.algorithms.base import BaseAlgorithm
from src.envs.inventory.simulator import Simulator

class PolicyLoss(nn.Module):
    """
    Loss that returns the sum of the rewards.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, observation, action, reward):
        return reward.sum()

    # def forward(self, tensordict):
    #     # Extract from tensordict
    #     total_cost = tensordict["total_cost"]
        
    #     # Compute HDPO loss using pathwise derivatives
    #     loss = total_cost.mean()  # Example computation for loss
    #     metrics = {"average_cost": total_cost.mean().item()}  # Example metric
    #     return {"loss": loss, "metrics": metrics}