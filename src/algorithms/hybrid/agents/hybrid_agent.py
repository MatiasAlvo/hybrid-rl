import torch
import torch.nn.functional as F
from src.algorithms.common.agents.base_agent import BaseAgent
from src.algorithms.common.policies.policy import HybridPolicy, NeuralNetworkCreator
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
        
        return NeuralNetworkCreator().get_architecture(policy_params['policy_network']['name'])(policy_params, device=self.device)
    
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
        
        # print(f'raw_outputs: {raw_outputs["discrete"][0]}')
        # Process network output
        action_dict = self.feature_registry.process_network_output(raw_outputs, argmax=False, sample=True)
        # print(f'average discrete_probs: {action_dict["discrete_probs"].mean(dim=0)}')
        # Get value if value network exists
        value = self.value_net(observation) if self.value_net is not None else None

        # I want to override action_dict['feature_actions']['total_action'] for testing purposes
        # I am implementing a base stock policy with a cap in orders. therefore, first get total inventory across time
        # get difference between total inventory and base stock level (in this case is 19)
        # then, get difference between total action and difference
        # then, set total action to total action - difference
        # finally, cap the action at 6
        # total_inventory = observation['store_inventories'].sum(dim=2)
        # base_stock_level = 19
        # difference = base_stock_level - total_inventory
        # total_action = torch.clip(difference, max=6)
        # action_dict['feature_actions']['total_action'] = total_action
        # print(f'discrete_probs: {action_dict["discrete_probs"][0]}')
        # print(f'total_action: {action_dict["feature_actions"]["total_action"][0]}')
        # print sum of inventory
        # print(f'sum of inventory: {observation["store_inventories"].sum(dim=2)[0]}')
        # print()
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs
        }

    def get_entropy(self, logits):
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.entropy()
    
    def get_logits_value_and_entropy(self, processed_observation, actions):
        """Get logits for specific actions, value, and entropy"""
        raw_outputs = self.policy(processed_observation, process_state=False)
        
        # raw_outputs['discrete'] shape is [batch_size, n_stores, n_actions]
        # actions shape is [batch_size]
        # Need to reshape actions to match the logits dimension
        actions = actions.view(-1, 1, 1).expand(-1, raw_outputs['discrete'].size(1), 1)
        
        # Gather logits for the specific actions that were taken
        logits = raw_outputs['discrete'].gather(-1, actions).squeeze(-1)
        
        value = self.value_net(processed_observation.detach(), process_state=False) if self.value_net else None
        entropy = self.get_entropy(raw_outputs['discrete'])
        
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

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to remove initialization hooks before loading"""
        # Remove hooks from policy network before loading
        if hasattr(self.policy, 'remove_lazy_init_hooks'):
            print("Removing lazy initialization hooks from policy")
            self.policy.remove_lazy_init_hooks()
        
        # Load state dict
        return super().load_state_dict(state_dict, strict)