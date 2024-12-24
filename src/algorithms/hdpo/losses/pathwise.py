from .base import InventoryLoss
import torch
from tensordict.tensordict import TensorDict

class HDPOLoss(InventoryLoss):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, tensordict):
        # Extract data
        actions = tensordict["action"]
        rewards = tensordict["reward"]
        
        # Your existing pathwise loss computation
        loss = self.compute_pathwise_loss(actions, rewards)
        
        return TensorDict({
            "loss": loss,
            "metrics": {
                "pathwise_loss": loss.detach(),
            }
        }, batch_size=[])