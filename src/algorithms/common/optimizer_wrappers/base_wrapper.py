import torch
import torch.nn.functional as F

class BaseOptimizerWrapper:
    """Base optimizer wrapper with all possible optimization methods"""
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def compute_policy_loss(self, log_probs, advantages):
        """Policy gradient loss"""
        return -(log_probs * advantages).mean()
    
    def compute_value_loss(self, values, returns):
        """Value function loss"""
        return F.mse_loss(values, returns)
    
    def compute_entropy_loss(self, distribution):
        """Entropy bonus loss"""
        return -distribution.entropy().mean()
    
    def clip_gradients(self, max_grad_norm):
        """Gradient clipping utility"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
    
    def optimize(self, trajectory_data, rewards):
        """Each algorithm implements its specific optimization logic"""
        raise NotImplementedError 