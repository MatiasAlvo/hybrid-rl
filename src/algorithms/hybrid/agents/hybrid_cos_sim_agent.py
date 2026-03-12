import torch

from src.algorithms.hybrid.agents.hybrid_agent import GaussianPPOAgent


class HybridCosSimAgent(GaussianPPOAgent):
    """
    Hybrid agent variant for cosine-similarity analysis.
    - Uses GaussianPPOAgent logic for stochastic continuous actions.
    - Fixes std to a constant (default 1.0) and freezes it.
    - Keeps Hybrid-style required losses (policy + value + pathwise + entropy).
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        agent_params = config.get('agent_params', {})
        fixed_std_value = agent_params.get('fixed_std', 1.0)
        if isinstance(fixed_std_value, bool):
            fixed_std_value = 1.0 if fixed_std_value else 1.0
        
        agent_params['fixed_std'] = True
        config['agent_params'] = agent_params
        self.fixed_std_value = float(fixed_std_value)
        
        super().__init__(config, feature_registry, device)
        
        if hasattr(self, 'log_std') and self.log_std is not None:
            with torch.no_grad():
                self.log_std.fill_(torch.log(torch.tensor(self.fixed_std_value, device=self.log_std.device)))
            self.log_std.requires_grad_(False)
    
    def _get_required_losses(self):
        """Enable hybrid losses for cosine-gradient analysis."""
        return {
            'policy_gradient': True,
            'value': True,
            'pathwise': True,
            'entropy': True
        }

    def load_state_dict(self, state_dict, strict=True):
        """
        Allow loading HybridAgent checkpoints that don't include log_std.
        """
        if 'log_std' not in state_dict and hasattr(self, 'log_std'):
            strict = False
        return super().load_state_dict(state_dict, strict)
