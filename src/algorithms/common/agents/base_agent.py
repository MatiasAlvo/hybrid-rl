import torch
import torch.nn.functional as F
from src.algorithms.common.policies.policy import NeuralNetworkCreator
from src.algorithms.common.values.value_network import ValueNetwork
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_registry = feature_registry
        
        # Initialize networks
        self.policy = self._init_policy(config)
        self.value_net = self._init_value_network(config) if config['nn_params'].get('value_network', {}).get('enabled', False) else None
    
    def _init_policy(self, config):
        """Initialize the policy network. Override in subclasses."""
        raise NotImplementedError
    
    def _init_value_network(self, config):
        """Initialize the value network if enabled"""
        if not config['nn_params'].get('value_network', {}).get('enabled', False):
            return None
        
        value_params = config['nn_params']['value_network']
        value_params['input_size'] = self.feature_registry.get_network_dimensions()['input_size']
        return ValueNetwork(value_params, device=self.device)
    
    def parameters(self):
        """Return parameters for optimization"""
        if self.value_net is not None:
            return list(self.policy.parameters()) + list(self.value_net.parameters())
        return self.policy.parameters()
    
    def forward(self, observation):

        raise NotImplementedError

    def trainable(self):
        """Return whether the agent is trainable"""
        return self.policy.trainable
