import torch
import torch.nn.functional as F
from src.algorithms.common.agents.base_agent import BaseAgent
from src.algorithms.common.policies.policy import HybridPolicy
from src.algorithms.common.values.value_network import ValueNetwork

class HybridAgent(BaseAgent):
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        self.policy = self._init_policy(config)
        self.value_net = self._init_value(config)
        
    def _init_policy(self, config):
        """Initialize the policy network"""
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # Update input/output sizes in config
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        policy_params['policy_network']['heads']['discrete']['size'] = network_dims['n_discrete']
        policy_params['policy_network']['heads']['continuous']['size'] = network_dims['n_continuous']
        
        return HybridPolicy(policy_params, device=self.device)
    
    def parameters(self):
        """Return parameters for optimization"""
        return list(self.policy.parameters()) + list(self.value_net.parameters()) if self.value_net else self.policy.parameters()

    def forward(self, observation):
        """Forward pass through the agent"""
        # Get raw outputs from policy
        raw_outputs = self.policy(observation)
        
        # Debug raw outputs
        if isinstance(raw_outputs, dict) and 'discrete' in raw_outputs:
            discrete_logits = raw_outputs['discrete']
            if torch.isnan(discrete_logits).any() or torch.isinf(discrete_logits).any():
                print("Warning: NaN or Inf in discrete logits from policy")
                print("Discrete logits stats:", 
                      f"range [{discrete_logits.min().item():.3f}, {discrete_logits.max().item():.3f}], "
                      f"mean {discrete_logits.mean().item():.3f}")
        
        # Process network output
        action_dict = self.feature_registry.process_network_output(raw_outputs, argmax=False, sample=True)
        
        # Get value if value network exists
        value = self.value_net(observation) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs
        }

    def get_entropy(self, logits):
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.entropy()
    
    def get_logits_value_and_entropy(self, processed_observation):
        """Get logits, value, and entropy
        Note that processed_observation is already processed by the feature registry, so we flag it as False
        """
        
        logits = self.policy(processed_observation, process_state=False)['discrete']
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        entropy = self.get_entropy(logits)
        return logits, value, entropy
    
    # def transform_outputs(self, raw_outputs):
    #     """Transform network outputs to action space"""
    #     if not self.feature_registry:
    #         return {'simulator_actions': raw_outputs}
        
    #     # Apply softmax to discrete outputs
    #     discrete_probs = F.softmax(raw_outputs['discrete'], dim=-1)
        
    #     # Scale continuous outputs
    #     continuous_values = self.feature_registry.scale_continuous_outputs(
    #         raw_outputs['continuous']
    #     )
        
    #     # Convert to simulator actions
    #     simulator_actions = self.feature_registry.convert_network_output_to_simulator_action(
    #         discrete_probs,
    #         continuous_values
    #     )

    #     print(f"raw_outputs['discrete']: {raw_outputs['discrete'][0]}")
    #     print(f"discrete_probs: {discrete_probs[0]}")
    #     print(f"raw_outputs['continuous']: {raw_outputs['continuous'][0]}")
    #     print(f"continuous_values: {continuous_values[0]}") 
    #     print(f"simulator_actions: {simulator_actions[0]}")
        
    #     return {
    #         'simulator_actions': simulator_actions,
    #         'processed_outputs': {
    #             'discrete': discrete_probs,
    #             'continuous': continuous_values
    #         },
    #         'raw_outputs': raw_outputs
        # }
    
    def _init_value(self, config):
        """Initialize the value network if enabled in config"""
        if not config['nn_params'].get('value_network', {}).get('enabled', False):
            return None
        
        value_params = config['nn_params']['value_network']
        
        # Create value network with observation keys attribute
        value_net = ValueNetwork(
            config={
                'hidden_layers': value_params['hidden_layers'],
                'activation': value_params['activation'],
                'dropout': value_params.get('dropout', 0.0),
                'batch_norm': value_params.get('batch_norm', False)
            },
            device=self.device
        )
        
        # Set observation keys attribute
        value_net.observation_keys = value_params.get('observation_keys', ['store_inventories'])
        
        return value_net