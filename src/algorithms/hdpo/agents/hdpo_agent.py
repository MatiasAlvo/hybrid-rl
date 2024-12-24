import torch
import torch.nn.functional as F
from src.algorithms.common.agents.base_agent import BaseAgent

class HDPOAgent(BaseAgent):
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
    
    def _init_policy(self, config):
        """Initialize the policy network"""
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # Update config for continuous outputs
        policy_params = config['nn_params']['policy_network']
        policy_params['heads']['continuous']['size'] = network_dims['n_continuous']
        policy_params['heads']['continuous']['enabled'] = True
        
        # Optionally enable discrete head for future Gumbel-Softmax implementation
        if network_dims.get('n_discrete'):
            policy_params['heads']['discrete']['size'] = network_dims['n_discrete']
            policy_params['heads']['discrete']['enabled'] = True
        
        # Create policy network
        from src.algorithms.common.policies.policy import HybridPolicy
        return HybridPolicy(policy_params, device=self.device)
    
    def transform_outputs(self, raw_outputs):
        """Transform network outputs to action space"""
        if not self.feature_registry:
            return {'simulator_actions': raw_outputs}
        
        transformed = {}
        
        # Handle continuous outputs
        continuous_values = self.feature_registry.scale_continuous_outputs(
            raw_outputs['continuous']
        )
        transformed['continuous'] = continuous_values
        
        # Handle discrete outputs if present (for future Gumbel-Softmax)
        if 'discrete' in raw_outputs:
            discrete_probs = F.softmax(raw_outputs['discrete'], dim=-1)
            transformed['discrete'] = discrete_probs
        
        # Convert to simulator actions
        simulator_actions = self.feature_registry.convert_network_output_to_simulator_action(
            transformed.get('discrete'),
            transformed['continuous']
        )
        
        return {
            'simulator_actions': simulator_actions,
            'probabilities': transformed,
            'raw_outputs': raw_outputs
        } 