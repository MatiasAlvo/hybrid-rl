from src.algorithms.common.optimizer_wrappers.base_wrapper import BaseOptimizerWrapper

class HDPOWrapper(BaseOptimizerWrapper):
    """HDPO optimizer wrapper - uses only policy gradient loss"""
    def __init__(self, model, optimizer, gradient_clip=None, weight_decay=0.0, device='cpu'):
        super().__init__(model, optimizer, device)
        self.gradient_clip = gradient_clip
        
    def optimize(self, trajectory_data, rewards):
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = -(trajectory_data['raw_outputs']['continuous'] * rewards).mean()
        
        # Backward pass
        loss.backward()
        
        # Optional gradient clipping
        if self.gradient_clip is not None:
            self.clip_gradients(self.gradient_clip)
        
        # Optimization step
        self.optimizer.step()
        
        return loss.item() 