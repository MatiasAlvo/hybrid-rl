import torch
import torch.nn.functional as F
import copy
from src.algorithms.common.policies.policy import HybridPolicy, NeuralNetworkCreator
# from src.algorithms.common.policies.policy import HybridPolicy, NeuralNetworkCreator, ContinuousPolicy
from src.algorithms.common.values.value_network import ValueNetwork
from src.algorithms.common.values.q_network import QNetwork
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from scipy.linalg import solve_discrete_are

class BaseAgent(nn.Module):
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_registry = feature_registry
        self.policy_config = config['nn_params']['policy_network']
        self.value_config = config['nn_params']['value_network']
        
        # Flag to indicate if agent has temperature parameter
        self.has_temperature = False
        
        # Initialize networks
        self.policy = self._init_policy(config)
        self.value_net = self._init_value_network(config) if config['nn_params'].get('value_network', {}).get('enabled', False) else None
        
        # Define which loss components this agent requires
        self.required_losses = self._get_required_losses()
    
    def _get_required_losses(self):
        """Define which loss components this agent requires. Override in subclasses."""
        return {
            'policy_gradient': False,  # PPO-style policy gradient loss
            'value': False,            # Value function loss
            'pathwise': False,         # Pathwise derivative loss
            'entropy': False           # Entropy loss
        }
    
    def _init_policy(self, config):
        """Initialize the policy network. Override in subclasses."""
        raise NotImplementedError
    
    def _init_value_network(self, config):
        """Initialize the value network if enabled"""
        if not config['nn_params'].get('value_network', {}).get('enabled', False):
            return None
        
        value_params = config['nn_params']['value_network']
        value_params['input_size'] = self.feature_registry.get_network_dimensions()['input_size']
        
        # Create value network 
        value_net = ValueNetwork(value_params, device=self.device)
        
        # Ensure observation_keys is properly set
        if 'observation_keys' in value_params:
            value_net.observation_keys = value_params['observation_keys']
        else:
            print("Warning: No observation_keys found in value network config, falling back to policy network observation keys")
            # Fall back to policy network observation keys for consistency
            value_net.observation_keys = self.policy.observation_keys
            
        return value_net
    
    def parameters(self):
        """Return parameters for optimization"""
        if self.value_net is not None:
            return list(self.policy.parameters()) + list(self.value_net.parameters())
        return self.policy.parameters()
    
    def forward(self, observation, train=True):
        """Forward pass through the agent. Override in subclasses."""
        raise NotImplementedError

    def trainable(self):
        """Return whether the agent is trainable"""
        return self.policy.trainable
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices, continuous_samples=None, detach_continuous=False):
        """Get logits for specific actions, value, and entropy. Override in subclasses."""
        raise NotImplementedError
    
    def get_discrete_log_probs(self, discrete_logits, discrete_action_indices):
        """
        Compute log probabilities for discrete actions using Categorical distribution
        
        Args:
            discrete_logits: Raw logits from policy [batch, n_discrete] or [batch, n_stores, n_discrete]
            discrete_action_indices: Indices of actions taken [batch] or [batch, n_stores]
            
        Returns:
            discrete_log_probs: Log probabilities of the taken actions
        """
        # Reshape logits to 2D for Categorical (batch_size*n_stores, n_discrete)
        original_shape = discrete_logits.shape
        reshaped_logits = discrete_logits.reshape(-1, discrete_logits.size(-1))
        
        # Create categorical distribution
        distribution = torch.distributions.Categorical(logits=reshaped_logits)
        
        # Reshape action indices to match reshaped logits
        reshaped_action_indices = discrete_action_indices.reshape(-1)
        
        # Get log probabilities using the distribution (this applies normalization)
        log_probs = distribution.log_prob(reshaped_action_indices)
        
        # Reshape back to original format
        log_probs = log_probs.reshape(original_shape[:-1])
        
        return log_probs
    
    def get_continuous_log_probs(self, continuous_mean, continuous_std, continuous_samples, discrete_action_indices=None):
        normal_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        continuous_log_probs = normal_dist.log_prob(continuous_samples)

        if discrete_action_indices is not None:
            # Treat [N] and [N,1] as a shared discrete action across stores
            if discrete_action_indices.dim() == 1:
                discrete_action_indices = discrete_action_indices.unsqueeze(1)
            if discrete_action_indices.dim() == 2 and discrete_action_indices.size(1) == 1:
                discrete_action_indices = discrete_action_indices.expand(-1, continuous_log_probs.size(1))

            gather_indices = discrete_action_indices.unsqueeze(-1)
            continuous_log_probs = continuous_log_probs.gather(-1, gather_indices).squeeze(-1)

        if continuous_log_probs.dim() > 2:
            continuous_log_probs = continuous_log_probs.sum(dim=-1)

        return continuous_log_probs
    
    def get_discrete_one_hot(self, discrete_logits, discrete_action_indices):
        """
        Create one-hot encoding for discrete actions
        
        Args:
            discrete_logits: Raw logits from policy [batch, n_discrete] or [batch, n_stores, n_discrete]
            discrete_action_indices: Indices of actions taken [batch] or [batch, n_stores]
            
        Returns:
            discrete_one_hot: One-hot encoded discrete actions with same shape as discrete_logits
        """
        # Reshape action indices to match logits dimensions
        actions = discrete_action_indices.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1] or [batch, n_stores, 1, 1]
        
        # Create one-hot encoding
        discrete_one_hot = torch.zeros_like(discrete_logits)
        discrete_one_hot.scatter_(-1, actions, 1)
        
        # Remove the extra dimension if needed
        if discrete_one_hot.dim() > discrete_logits.dim():
            discrete_one_hot = discrete_one_hot.squeeze(1)
        
        return discrete_one_hot
    
    def calculate_log_probs(self, discrete_logits, discrete_action_indices, 
                          selected_continuous_mean=None, continuous_std=None, 
                          raw_continuous_samples=None):
        """
        Calculate log probabilities for PPO training
        
        Args:
            discrete_logits: Raw discrete logits from policy
            discrete_action_indices: Indices of discrete actions taken
            selected_continuous_mean: Mean of continuous actions (optional)
            continuous_std: Standard deviation of continuous actions (optional)
            raw_continuous_samples: All sampled continuous actions for all discrete actions (optional)
            
        Returns:
            total_log_probs: Combined log probabilities of discrete and continuous actions
        """
        # Calculate discrete log probabilities
        discrete_log_probs = self.get_discrete_log_probs(discrete_logits, discrete_action_indices)
        
        # Initialize total log probabilities with discrete part
        total_log_probs = discrete_log_probs
        
        # Add continuous log probabilities if continuous components are provided
        if (selected_continuous_mean is not None and 
            continuous_std is not None and 
            raw_continuous_samples is not None):
            
            # Get the continuous samples for the selected discrete action
            actions = discrete_action_indices.unsqueeze(-1)  # [batch, 1, 1]
            selected_continuous_samples = raw_continuous_samples.gather(-1, actions)  # [batch, 1, 1]
            selected_continuous_samples = selected_continuous_samples.squeeze(-1)  # [batch, 1]
            
            continuous_log_probs = self.get_continuous_log_probs(
                selected_continuous_mean, 
                continuous_std, 
                selected_continuous_samples
            )
            # Combine discrete and continuous log probabilities
            total_log_probs = discrete_log_probs + continuous_log_probs
        
        return total_log_probs
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to remove initialization hooks before loading"""
        # Remove hooks from policy network before loading
        if hasattr(self.policy, 'remove_lazy_init_hooks'):
            print("Removing lazy initialization hooks from policy")
            self.policy.remove_lazy_init_hooks()
        
        # Load state dict
        return super().load_state_dict(state_dict, strict)

    def update_temperature(self):
        """Update temperature with decay schedule"""
        if hasattr(self, 'temperature') and hasattr(self, 'temperature_decay') and hasattr(self, 'min_temperature'):
            self.temperature = max(self.temperature * self.temperature_decay, self.min_temperature)
            return self.temperature
        return None

    def _init_factored_policy(self, config, continuous_size=None):
        """Initialize a factored policy network with separate discrete and continuous heads
        
        Args:
            config: Configuration dictionary
            continuous_size: Size of continuous head output. If None, determined by fixed_std parameter.
        
        Returns:
            Initialized factored policy network
        """
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # Update config
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        policy_params['policy_network']['heads']['discrete']['size'] = network_dims['n_discrete']
        
        # Determine continuous head size based on fixed_std parameter
        if continuous_size is not None:
            # Use explicitly provided size
            policy_params['policy_network']['heads']['continuous']['size'] = continuous_size
        else:
            # Determine size based on fixed_std parameter
            fixed_std = config.get('agent_params', {}).get('fixed_std', True)
            if fixed_std:
                # Fixed std: output mean per store
                policy_params['policy_network']['heads']['continuous']['size'] = self.feature_registry.n_stores
            else:
                # State-dependent std shared across products: mean per store + scalar log_std
                policy_params['policy_network']['heads']['continuous']['size'] = self.feature_registry.n_stores + 1
        
        # Provide store/range metadata for policy outputs
        policy_params['policy_network']['n_stores'] = self.feature_registry.n_stores
        policy_params['policy_network']['n_sub_ranges'] = self.feature_registry.n_sub_ranges
        
        # Create new policy network from the FactoredPolicy class
        policy_class = NeuralNetworkCreator().get_architecture("factored_policy")
        return policy_class(policy_params, device=self.device)

class HybridAgent(BaseAgent):
    """
    Original hybrid agent with:
    - Discrete actions: score-function gradient (PPO)
    - Continuous actions: pathwise + score-function
    - Requires value network for PPO
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        # No additional initialization needed here as the value_net is initialized in BaseAgent
        
    def _init_policy(self, config, random_continuous=False):
        """Initialize the policy network"""
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # Update input/output sizes in config
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        policy_params['policy_network']['heads']['discrete']['size'] = network_dims['n_discrete']
        
        # If using random continuous actions (Gaussian), double the continuous size
        # to accommodate both mean and log_std
        if random_continuous:
            continuous_size = network_dims['n_continuous'] * 2
            print(f"Using random continuous actions - doubling continuous head size to {continuous_size}")
        else:
            continuous_size = network_dims['n_continuous']
            
        policy_params['policy_network']['heads']['continuous']['size'] = continuous_size
        print(f"continuous size: {continuous_size}")
        
        return NeuralNetworkCreator().get_architecture(policy_params['policy_network']['name'])(policy_params, device=self.device)

    def forward(self, observation, train=True):
        """Forward pass through the agent using the new processing functions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Debug raw outputs
        debug_here = False
        if debug_here:
            if isinstance(raw_outputs, dict) and 'discrete' in raw_outputs:
                discrete_logits = raw_outputs['discrete']
                if torch.isnan(discrete_logits).any() or torch.isinf(discrete_logits).any():
                    print("Warning: NaN or Inf in discrete logits from policy")
                    print("Discrete logits stats:", 
                        f"range [{discrete_logits.min().item():.3f}, {discrete_logits.max().item():.3f}], "
                        f"mean {discrete_logits.mean().item():.3f}")
        
        # Process discrete outputs
        discrete_output = self.feature_registry.process_discrete_output(
            # raw_outputs['discrete'].detach(),
            raw_outputs['discrete'],
            # argmax=False,  # Use argmax for inference, sample for training
            argmax=not train,  # Use argmax for inference, sample for training
            sample=True,      # Sample during training
            straight_through=False
        )

        # # detach every entry in discrete_output
        # for key in discrete_output:
        #     discrete_output[key] = discrete_output[key].detach()
        
        # Process continuous outputs
        apply_scaling = not self.policy_config.get('disable_continuous_scaling', False)
        continuous_output = self.feature_registry.process_continuous_output(
            raw_outputs.get('continuous'),
            discrete_action_indices=discrete_output['discrete_action_indices'],
            continuous_mean=raw_outputs.get('continuous_mean'),
            continuous_log_std=raw_outputs.get('continuous_log_std'),
            random_continuous=False,  # Default to deterministic continuous actions
            observations=observation,
            apply_scaling=apply_scaling
        )
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': discrete_output['log_probs'],
            'continuous_values': continuous_output['continuous_values'],
            'raw_continuous_samples': continuous_output['raw_continuous_samples'],
            'feature_actions': feature_actions
        }
        
        # Add continuous log probs if available
        if continuous_output['continuous_log_probs'] is not None:
            action_dict['continuous_log_probs'] = continuous_output['continuous_log_probs']
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }
    
    def get_entropy(self, logits):
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.entropy()
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None, detach_continuous=False):
        """
        Get logits, value, and entropy for PPO
        """
        # Get policy outputs
        raw_outputs = self.policy(processed_observation, process_state=False)
        
        # Calculate discrete log probabilities using BaseAgent helper function
        log_probs = self.get_discrete_log_probs(raw_outputs['discrete'], discrete_action_indices)
        
        value = self.value_net(processed_observation.detach(), process_state=False) if self.value_net else None
        entropy = self.get_entropy(raw_outputs['discrete'])
        
        return log_probs, value, entropy
    
    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': True,   # For discrete actions
            'value': True,             # For PPO advantage estimation
            'pathwise': True,          # For continuous actions
            'entropy': True            # For exploration
        }


class LearnedCriticPathwiseAgent(HybridAgent):
    """
    Hybrid agent ablation where continuous pathwise gradients are routed through
    a learned Q(s, a) critic instead of true simulator gradients.
    """

    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)

        q_params = copy.deepcopy(config['nn_params'].get('q_network', {}))
        q_params['input_size'] = self.feature_registry.get_network_dimensions()['input_size']
        q_params['action_size'] = self.feature_registry.n_stores
        self.q_net = QNetwork(q_params, device=device)

        self.q_net_target = copy.deepcopy(self.q_net)
        for param in self.q_net_target.parameters():
            param.requires_grad = False

        agent_params = config.get('agent_params', {})
        self.tau = float(agent_params.get('tau', 0.005))
        self.gamma = float(agent_params.get('gamma', 0.99))

    def _get_required_losses(self):
        return {
            'policy_gradient': True,
            'value': True,
            'pathwise': False,
            'entropy': True,
            'learned_critic': True
        }

    def soft_update_target(self):
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def parameters(self):
        params = list(self.policy.parameters())
        if self.value_net is not None:
            params += list(self.value_net.parameters())
        params += list(self.q_net.parameters())
        return params

class AlternateHybridAgent(HybridAgent):

    pass

class FixedDiscreteHybridAgent(HybridAgent):

    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': False,   # For discrete actions
            'value': False,             # For PPO advantage estimation
            'pathwise': True,          # For continuous actions
            'entropy': False            # For exploration
        }

    def base_stock_ordering_policy(self, observation):
        """
        Vectorized base-stock ordering policy based on the plot.
        
        Orders (purple/red dots) occur when:
        - x1 <= -2, OR
        - x2 <= -2, OR
        - x1 = -1 AND x2 = -1
        
        Args:
            observation: dict containing 'store_inventories' of shape [batch, 2, 1]
        
        Returns:
            dict with:
                'discrete_probs': shape [batch, 1, 2] where [..., 0] = prob of no-order and [..., 1] = prob of order
                'discrete_action_indices': shape [batch, 1] with action indices (0 = no-order, 1 = order)
        """
        # Extract and squeeze to [batch, 2]
        states = observation['store_inventories'].squeeze(-1)  # [batch, 2]
        
        x1 = states[:, 0]  # [batch]
        x2 = states[:, 1]  # [batch]
        
        # Define ordering conditions
        # Order when x1 <= -2 OR x2 <= -2 OR (x1 = -1 AND x2 = -1)
        condition_x1 = x1 <= -2
        condition_x2 = x2 <= -2
        condition_special = (x1 == -1) & (x2 == -1)
        
        should_order = condition_x1 | condition_x2 | condition_special  # [batch]
        
        # Create discrete_probs tensor [batch, 1, 2]
        # Format: [..., 0] = no-order, [..., 1] = order
        discrete_probs = torch.zeros(states.shape[0], 1, 2, 
                                    dtype=states.dtype, 
                                    device=states.device)
        
        # Set probabilities: [1, 0] for no-order, [0, 1] for order
        discrete_probs[should_order, 0, 0] = 0  # no-order prob
        discrete_probs[should_order, 0, 1] = 1  # order prob
        
        discrete_probs[~should_order, 0, 0] = 1  # no-order prob
        discrete_probs[~should_order, 0, 1] = 0  # order prob
        
        # Create discrete_action_indices [batch, 1]
        # 0 = no-order, 1 = order
        discrete_action_indices = should_order.long().unsqueeze(1)  # [batch, 1]
        
        discrete_output = {
            'discrete_probs': discrete_probs,
            'discrete_action_indices': discrete_action_indices
        }
        
        return discrete_output
    
    def base_stock_continuous_ordering_policy(self, observation, base_stock_levels):
        """
        Vectorized base-stock ordering policy that computes continuous order quantities.
        
        Args:
            observation: dict containing 'store_inventories' of shape
                [batch, n_items, 1] or [batch, n_items, lead_time]
            base_stock_levels: list or tensor of length n_items, containing the base-stock level for each item
        
        Returns:
            dict with 'continuous_values' of shape [batch, 2, n_items]
                where [..., 0, :] = 0 (no continuous component for no-order action)
                and [..., 1, :] = order quantities to reach base_stock_levels
        """
        # Extract inventories; handle optional lead-time dimension
        states = observation['store_inventories'].sum(dim=-1)
        
        batch_size = states.shape[0]
        n_items = states.shape[1]
        
        # Convert base_stock_levels to tensor if needed
        if not isinstance(base_stock_levels, torch.Tensor):
            base_stock_levels = torch.tensor(
                base_stock_levels,
                dtype=states.dtype,
                device=states.device
            )
        
        # Allow either per-item [n_items] or per-sample [batch, n_items]
        if base_stock_levels.dim() == 1:
            base_stock_levels = base_stock_levels.view(1, n_items).expand(batch_size, -1)
        elif base_stock_levels.dim() == 2:
            if base_stock_levels.shape[0] != batch_size or base_stock_levels.shape[1] != n_items:
                raise ValueError(
                    f"base_stock_levels has shape {tuple(base_stock_levels.shape)}, "
                    f"expected ({batch_size}, {n_items})."
                )
        else:
            raise ValueError(
                f"base_stock_levels must have shape [n_items] or [batch, n_items], "
                f"got {tuple(base_stock_levels.shape)}."
            )
        
        # Compute order quantities: max(0, S - x) for each item
        order_quantities = torch.clamp(base_stock_levels - states, min=0)  # [batch, n_items]
        
        # Build continuous values without in-place assignment to preserve gradients
        zeros = torch.zeros_like(order_quantities)
        # Shape: [batch, n_items, 2] where index 0 = no-order, index 1 = order qty
        continuous_values = torch.stack([zeros, order_quantities], dim=-1)
        
        continuous_output = {
            'continuous_values': continuous_values
        }
        
        return continuous_output

    def forward(self, observation, train=True):
        """Forward pass through the agent using the new processing functions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        discrete_output = self.base_stock_ordering_policy(observation)
        
        # Process continuous outputs
        continuous_output = self.feature_registry.process_continuous_output(
            raw_outputs.get('continuous'),
            discrete_action_indices=discrete_output['discrete_action_indices'],
            continuous_mean=raw_outputs.get('continuous_mean'),
            continuous_log_std=raw_outputs.get('continuous_log_std'),
            random_continuous=False,  # Default to deterministic continuous actions
            observations=observation
        )
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': None,
            'continuous_values': continuous_output['continuous_values'],
            'feature_actions': feature_actions
        }
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }

class FixedContinuousHybridAgent(FixedDiscreteHybridAgent):

    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        agent_params = config.get('agent_params', {})
        # Fixed base-stock level for all store/product pairs.
        self.fixed_base_stock_level = agent_params.get('fixed_base_stock_level', 62)

    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': True,   # For discrete actions
            'value': True,             # For PPO advantage estimation
            'pathwise': False,          # For continuous actions
            'entropy': True            # For exploration
        }
    
    def forward(self, observation, train=True):
        """Forward pass through the agent using the new processing functions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Process discrete outputs
        discrete_output = self.feature_registry.process_discrete_output(
            # raw_outputs['discrete'].detach(),
            raw_outputs['discrete'],
            # argmax=False,  # Use argmax for inference, sample for training
            argmax=not train,  # Use argmax for inference, sample for training
            sample=True,      # Sample during training
            straight_through=False
        )
        
        # Process continuous outputs
        n_items = observation['store_inventories'].shape[1]
        base_stock_levels = [self.fixed_base_stock_level] * n_items
        continuous_output = self.base_stock_continuous_ordering_policy(
            observation,
            base_stock_levels
        )
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': discrete_output['log_probs'],
            'continuous_values': continuous_output['continuous_values'],
            'feature_actions': feature_actions
        }
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }

class SwitchedLqrPolicyAgent(BaseAgent):
    """
    Fixed switched LQR policy based on a finite set of P matrices (H^eps_k).
    Selects (mode, P) that minimizes x^T rho_i(P) x, then applies u = -K_i(P) x.
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        self.agent_params = config.get('agent_params', {})
        self._dummy_param = nn.Parameter(torch.zeros(1, device=self.device))
        self._load_lqr_matrices(config)
        self._load_policy_matrices()
        self._precompute_policy_terms()

    def _init_policy(self, config):
        policy = nn.Identity()
        policy.trainable = False
        policy.observation_keys = self.policy_config.get('observation_keys', ['store_inventories'])
        return policy

    def _get_required_losses(self):
        return {
            'policy_gradient': False,
            'value': False,
            'pathwise': False,
            'entropy': False
        }

    def _load_lqr_matrices(self, config):
        problem_params = config['scenario'].problem_params
        lqr_params = problem_params.get('lqr', {})
        missing = [key for key in ('A', 'B', 'Q', 'R') if key not in lqr_params]
        if missing:
            raise ValueError(f"Missing LQR parameters for SwitchedLqrPolicyAgent: {missing}.")

        self.lqr_A = torch.tensor(lqr_params['A'], device=self.device, dtype=torch.float32)
        self.lqr_B = torch.tensor(lqr_params['B'], device=self.device, dtype=torch.float32)
        self.lqr_Q = torch.tensor(lqr_params['Q'], device=self.device, dtype=torch.float32)
        self.lqr_R = torch.tensor(lqr_params['R'], device=self.device, dtype=torch.float32)

        self.n_modes, self.state_dim, _ = self.lqr_A.shape
        self.action_dim = self.lqr_B.shape[2]

    def _load_policy_matrices(self):
        default_P = [
            [[6.065, 1.206], [1.206, 1.905]],
            [[9.087, 3.235], [3.235, 2.348]],
            [[5.108, 1.266], [1.266, 1.935]],
            [[7.219, 2.561], [2.561, 2.107]],
        ]
        P_list = self.agent_params.get('lqr_policy_P_matrices', default_P)
        self.P_mats = torch.tensor(P_list, device=self.device, dtype=torch.float32)
        if self.P_mats.dim() != 3 or self.P_mats.shape[1:] != (self.state_dim, self.state_dim):
            raise ValueError(
                f"Expected P matrices of shape [n_P, {self.state_dim}, {self.state_dim}], "
                f"got {tuple(self.P_mats.shape)}."
            )

    def _precompute_policy_terms(self):
        n_P = self.P_mats.shape[0]
        rho_list = []
        K_list = []
        eye = torch.eye(self.action_dim, device=self.device, dtype=torch.float32)

        for mode_idx in range(self.n_modes):
            A = self.lqr_A[mode_idx]
            B = self.lqr_B[mode_idx]
            Q = self.lqr_Q[mode_idx]
            R = self.lqr_R[mode_idx]
            rho_mode = []
            K_mode = []
            for p_idx in range(n_P):
                P = self.P_mats[p_idx]
                BtP = B.transpose(0, 1) @ P
                inv_term = torch.linalg.inv(R + BtP @ B + 1e-8 * eye)
                rho = Q + A.transpose(0, 1) @ P @ A - A.transpose(0, 1) @ P @ B @ inv_term @ BtP @ A
                K = inv_term @ BtP @ A
                rho_mode.append(rho)
                K_mode.append(K)
            rho_list.append(torch.stack(rho_mode, dim=0))
            K_list.append(torch.stack(K_mode, dim=0))

        self.rho_mats = torch.stack(rho_list, dim=0)  # [n_modes, n_P, n_state, n_state]
        self.K_mats = torch.stack(K_list, dim=0)      # [n_modes, n_P, n_action, n_state]

    def forward(self, observation, train=True):
        x = observation['store_inventories']
        if x.dim() == 3:
            x = x[:, :, 0]
        if x.shape[1] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: expected {self.state_dim}, got {x.shape[1]}."
            )

        # Compute x^T rho_i(P) x for all modes and P
        scores = torch.einsum('bi,mkij,bj->bmk', x, self.rho_mats, x)
        n_P = self.P_mats.shape[0]
        flat_scores = scores.reshape(scores.shape[0], -1)
        best_flat_idx = torch.argmin(flat_scores, dim=-1)
        best_mode = (best_flat_idx // n_P).long()
        best_p = (best_flat_idx % n_P).long()

        # Log ratio of discrete action 0 and basic state stats
        ratio_action0 = (best_mode == 0).float().mean().item()
        inventory_sum = x.sum(dim=1).mean().item()
        current_period = observation.get('current_period')
        if torch.is_tensor(current_period):
            current_period = int(current_period.item())
        print(
            f"[SwitchedLqrPolicyAgent] period={current_period} "
            f"inventory_sum_mean={inventory_sum:.4f} "
            f"action0_ratio={ratio_action0:.4f}"
        )

        # Select gain and compute control u = -K x
        K_selected = self.K_mats[best_mode, best_p]  # [B, n_action, n_state]
        u = -torch.einsum('bij,bj->bi', K_selected, x)  # [B, n_action]

        # Build discrete one-hot probabilities [B, 1, n_modes]
        discrete_probs = torch.zeros(x.shape[0], 1, self.n_modes, device=self.device, dtype=x.dtype)
        discrete_probs.scatter_(2, best_mode.view(-1, 1, 1), 1.0)
        discrete_action_indices = best_mode.view(-1, 1)

        # Build continuous_values [B, n_stores, n_sub_ranges]
        n_stores = self.feature_registry.n_stores
        n_sub_ranges = self.feature_registry.n_sub_ranges
        u_full = torch.zeros(x.shape[0], n_stores, device=self.device, dtype=x.dtype)
        u_full[:, :self.action_dim] = u
        continuous_values = torch.zeros(x.shape[0], n_stores, n_sub_ranges, device=self.device, dtype=x.dtype)
        continuous_values.scatter_(2, best_mode.view(-1, 1, 1).expand(-1, n_stores, 1), u_full.unsqueeze(-1))

        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_probs,
            continuous_values
        )

        action_dict = {
            'discrete_probs': discrete_probs,
            'discrete_action_indices': discrete_action_indices,
            'log_probs': None,
            'continuous_values': continuous_values,
            'feature_actions': feature_actions
        }

        processed_obs = self.feature_registry.prepare_inputs(observation)
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': {},
            'vectorized_observation': processed_obs
        }

class FixedModeLqrRiccatiAgent(BaseAgent):
    """
    Fixed-mode LQR policy using the discrete-time Riccati solution for one mode.
    The discrete action is hardcoded to a single mode, and control is u = -Kx.
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        self.agent_params = config.get('agent_params', {})
        self._dummy_param = nn.Parameter(torch.zeros(1, device=self.device))
        self._load_lqr_matrices(config)
        self._load_fixed_mode()
        self._compute_lqr_gain()

    def _init_policy(self, config):
        policy = nn.Identity()
        policy.trainable = False
        policy.observation_keys = self.policy_config.get('observation_keys', ['store_inventories'])
        return policy

    def _get_required_losses(self):
        return {
            'policy_gradient': False,
            'value': False,
            'pathwise': False,
            'entropy': False
        }

    def _load_lqr_matrices(self, config):
        problem_params = config['scenario'].problem_params
        lqr_params = problem_params.get('lqr', {})
        missing = [key for key in ('A', 'B', 'Q', 'R') if key not in lqr_params]
        if missing:
            raise ValueError(f"Missing LQR parameters for FixedModeLqrRiccatiAgent: {missing}.")

        self.lqr_A = torch.tensor(lqr_params['A'], device=self.device, dtype=torch.float32)
        self.lqr_B = torch.tensor(lqr_params['B'], device=self.device, dtype=torch.float32)
        self.lqr_Q = torch.tensor(lqr_params['Q'], device=self.device, dtype=torch.float32)
        self.lqr_R = torch.tensor(lqr_params['R'], device=self.device, dtype=torch.float32)

        self.n_modes, self.state_dim, _ = self.lqr_A.shape
        self.action_dim = self.lqr_B.shape[2]

    def _load_fixed_mode(self):
        requested_mode = int(self.agent_params.get('fixed_lqr_mode', 0))
        if requested_mode < 0:
            raise ValueError(f"fixed_lqr_mode must be non-negative, got {requested_mode}.")
        # 0-based indexing: clip out-of-range sweep values to the last valid mode.
        self.fixed_mode = min(requested_mode, self.n_modes - 1)

    def _compute_lqr_gain(self):
        mode = self.fixed_mode
        A = self.lqr_A[mode].detach().cpu().numpy()
        B = self.lqr_B[mode].detach().cpu().numpy()
        Q = self.lqr_Q[mode].detach().cpu().numpy()
        R = self.lqr_R[mode].detach().cpu().numpy()

        P = solve_discrete_are(A, B, Q, R)
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)

        self.P = torch.tensor(P, device=self.device, dtype=torch.float32)
        self.K = torch.tensor(K, device=self.device, dtype=torch.float32)

    def forward(self, observation, train=True):
        x = observation['store_inventories']
        if x.dim() == 3:
            x = x[:, :, 0]
        if x.shape[1] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: expected {self.state_dim}, got {x.shape[1]}."
            )

        u = -torch.einsum('ij,bj->bi', self.K, x)

        batch_size = x.shape[0]
        n_stores = self.feature_registry.n_stores
        n_sub_ranges = self.feature_registry.n_sub_ranges
        if n_sub_ranges <= 0:
            raise ValueError(f"Expected n_sub_ranges > 0, got {n_sub_ranges}.")

        # Use fixed LQR mode for controller selection, but keep action tensors aligned
        # to range-manager dimensions (n_sub_ranges) to avoid invalid CUDA scatter indices.
        action_idx_val = min(self.fixed_mode, n_sub_ranges - 1)
        action_idx = torch.full((batch_size, 1), action_idx_val, device=self.device, dtype=torch.long)
        discrete_probs = torch.zeros(batch_size, 1, n_sub_ranges, device=self.device, dtype=x.dtype)
        discrete_probs.scatter_(2, action_idx.unsqueeze(-1), 1.0)

        u_full = torch.zeros(batch_size, n_stores, device=self.device, dtype=x.dtype)
        u_full[:, :self.action_dim] = u
        continuous_values = torch.zeros(batch_size, n_stores, n_sub_ranges, device=self.device, dtype=x.dtype)
        continuous_values.scatter_(2, action_idx.unsqueeze(1).expand(-1, n_stores, 1), u_full.unsqueeze(-1))

        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_probs,
            continuous_values
        )

        action_dict = {
            'discrete_probs': discrete_probs,
            'discrete_action_indices': action_idx,
            'log_probs': None,
            'continuous_values': continuous_values,
            'feature_actions': feature_actions
        }

        processed_obs = self.feature_registry.prepare_inputs(observation)
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': {},
            'vectorized_observation': processed_obs
        }

class TrainableBaseStockHybridAgent(FixedContinuousHybridAgent):

    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        agent_params = config.get('agent_params', {})
        self.base_stock_shared = agent_params.get('base_stock_shared', True)
        n_items = self.feature_registry.n_stores if self.feature_registry is not None else 1

        if self.base_stock_shared:
            initial_level = float(self.fixed_base_stock_level)
            # Train a single shared base-stock level (kept non-negative via softplus).
            self.base_stock_level = nn.Parameter(
                torch.tensor(initial_level, dtype=torch.float32, device=self.device)
            )
        else:
            initial_levels = agent_params.get('base_stock_levels', None)
            if initial_levels is None:
                initial_levels = [float(self.fixed_base_stock_level)] * n_items
            # Train per-store base-stock levels (kept non-negative via softplus).
            self.base_stock_levels = nn.Parameter(
                torch.tensor(initial_levels, dtype=torch.float32, device=self.device)
            )

    def _get_required_losses(self):
        """Train discrete PPO and pathwise continuous losses."""
        return {
            'policy_gradient': True,   # For discrete actions
            'value': True,             # For PPO advantage estimation
            'pathwise': True,          # For continuous actions
            'entropy': True            # For exploration
        }
    
    def forward(self, observation, train=True):
        """Forward pass with learned shared or per-store base-stock levels."""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Process discrete outputs
        discrete_output = self.feature_registry.process_discrete_output(
            raw_outputs['discrete'],
            argmax=not train,
            sample=True,
            straight_through=False
        )
        
        # Process continuous outputs using policy output as base-stock level(s)
        continuous_raw = raw_outputs.get('continuous')
        if continuous_raw is None:
            raise ValueError("Missing continuous outputs for base-stock policy.")

        # Reshape to [batch, n_stores, n_sub_ranges] when needed
        continuous_reshaped = self.feature_registry.reshape_continuous_output(continuous_raw)
        if continuous_reshaped is None:
            raise ValueError("Could not reshape continuous outputs for base-stock policy.")

        # Interpret the last sub-range as base-stock level per store
        if continuous_reshaped.dim() == 3:
            base_stock_levels = continuous_reshaped[..., -1]
        elif continuous_reshaped.dim() == 2:
            base_stock_levels = continuous_reshaped
        else:
            raise ValueError(
                f"Unexpected continuous output shape for base-stock policy: "
                f"{tuple(continuous_reshaped.shape)}"
            )
        base_stock_levels = F.softplus(base_stock_levels)
        continuous_output = self.base_stock_continuous_ordering_policy(
            observation,
            base_stock_levels
        )
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': discrete_output['log_probs'],
            'continuous_values': continuous_output['continuous_values'],
            'feature_actions': feature_actions
        }
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }

    def parameters(self):
        """Include trainable base-stock parameters in optimization."""
        params = super().parameters()
        if isinstance(params, list):
            param_list = params
        else:
            param_list = list(params)
        if self.base_stock_shared:
            param_list.append(self.base_stock_level)
        else:
            param_list.append(self.base_stock_levels)
        return param_list

class OptimalMultiItem(FixedDiscreteHybridAgent):

    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': False,   # For discrete actions
            'value': False,             # For PPO advantage estimation
            'pathwise': False,          # For continuous actions
            'entropy': False            # For exploration
        }
    
    # optimal cost for low variability setting in Ata 2025 (see Figure 2a)
    #Epoch 3: Train Loss = 1.2461, Dev Loss = 1.2421
    def forward(self, observation, train=True):
        """Forward pass through the agent using the new processing functions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        discrete_output = self.base_stock_ordering_policy(observation)
        
        # Process continuous outputs
        continuous_output = self.base_stock_continuous_ordering_policy(observation, [35, 19])
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': None,
            'continuous_values': continuous_output['continuous_values'],
            'feature_actions': feature_actions
        }
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }

class VarianceScalingAgent(FixedDiscreteHybridAgent):
         
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__(config, feature_registry, device)
        if self.value_net is not None:
            with torch.no_grad():
                self.value_net.head.weight.zero_()
                self.value_net.head.bias.zero_()
            print("Initialized value_net head to output zeros")
    
    # def value_net(self, observation, process_state=True):
    #     if process_state:
    #         observation = self.feature_registry.prepare_inputs(observation)
    #     else:
    #         observation = observation
        
    #     print(f'store inventories: {observation.shape}')
    #     return torch.zeros_like(observation[:, 0])
    #     # return torch.zeros_like(observation['store_inventories'][:, 0])

    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': True,   # For discrete actions
            'value': False,             # For PPO advantage estimation
            'pathwise': True,          # For continuous actions
            'entropy': False            # For exploration
        }
    
    def forward(self, observation, train=True):
        """Forward pass through the agent using the new processing functions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        
        
        # Process discrete outputs
        discrete_output = self.feature_registry.process_discrete_output(
            # raw_outputs['discrete'].detach(),
            raw_outputs['discrete'],
            # argmax=False,  # Use argmax for inference, sample for training
            argmax=not train,  # Use argmax for inference, sample for training
            sample=True,      # Sample during training
            straight_through=False
        )

        # # detach every entry in discrete_output
        # for key in discrete_output:
        #     discrete_output[key] = discrete_output[key].detach()
        
        # Process continuous outputs
        continuous_output = self.feature_registry.process_continuous_output(
            raw_outputs.get('continuous'),
            continuous_mean=raw_outputs.get('continuous_mean'),
            continuous_log_std=raw_outputs.get('continuous_log_std'),
            random_continuous=False,  # Default to deterministic continuous actions
            observations=observation
        )
        
        current_period = observation.get('current_period', None)
        
        if current_period.item() == 1:
            # print('using base stock policy')
            n_items = observation['store_inventories'].shape[1]
            base_stock_levels = [5.0] * n_items
            # base_stock_levels = [7.05] * n_items
            base_stock_output = self.base_stock_continuous_ordering_policy(
                observation,
                base_stock_levels
            )
            continuous_output['continuous_values'][..., 1] = (
                base_stock_output['continuous_values'][..., 1]
            )

        discrete_output['discrete_probs'][:, :, 0] = 0
        discrete_output['discrete_probs'][:, :, 1] = 1
        
        # Compute feature actions
        feature_actions = self.feature_registry.compute_feature_actions_from_outputs(
            discrete_output['discrete_probs'],
            continuous_output['continuous_values']
        )
        
        # Combine outputs into action dictionary
        action_dict = {
            'discrete_probs': discrete_output['discrete_probs'],
            'discrete_action_indices': discrete_output['discrete_action_indices'],
            'log_probs': discrete_output['log_probs'],
            'continuous_values': continuous_output['continuous_values'],
            # 'raw_continuous_samples': continuous_output['raw_continuous_samples'].detach().clone(),
            'feature_actions': feature_actions
        }
        
        # Add continuous log probs if available
        if continuous_output['continuous_log_probs'] is not None:
            action_dict['continuous_log_probs'] = continuous_output['continuous_log_probs']
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }

class FactoredHybridAgent(HybridAgent):
    """
    Factored hybrid agent with:
    - Discrete actions: score-function gradient (PPO)
    - Continuous actions: pathwise gradients
    - Requires value network for PPO
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # Set flag to identify this as a factored agent
        self.factored = True
        self.device = device
        super().__init__(config, feature_registry, device)
    
    def _init_policy(self, config):
        """Initialize policy with separate discrete and continuous networks"""
        return self._init_factored_policy(config)

    def forward(self, observation, train=True, process_state=True):
        """Forward pass: first sample discrete action, then get continuous mean"""
        # Process observation to get features if needed
        if process_state:
            processed_obs = self.feature_registry.prepare_inputs(observation)
        else:
            processed_obs = observation
            
        # First get discrete distribution
        discrete_logits = self.policy.get_discrete_output(processed_obs)
        
        # Sample discrete action
        discrete_distribution = torch.distributions.Categorical(logits=discrete_logits)
        discrete_action = discrete_distribution.sample()
        
        # Fix: Properly reshape discrete_action to match the expected dimensions
        # discrete_logits shape is [n_batch, 1, num_features]
        # discrete_action shape is [n_batch, 1]
        # Need to reshape discrete_action to [n_batch, 1, 1] for scatter_
        discrete_action_reshaped = discrete_action.unsqueeze(-1)
        discrete_one_hot = torch.zeros_like(discrete_logits).scatter_(-1, discrete_action_reshaped, 1)
        
        # Get continuous mean per store (conditioned on sampled discrete action)
        selected_continuous = self.policy.get_continuous_output(processed_obs, discrete_one_hot.squeeze(1))
        
        # Convert [batch, 1, n_stores] -> [batch, n_stores, n_sub_ranges]
        selected_continuous = selected_continuous.squeeze(1)
        expanded_continuous = selected_continuous.unsqueeze(-1).expand(
            -1, self.feature_registry.n_stores, self.feature_registry.n_sub_ranges
        )
        
        # Create raw_outputs dict in the format expected by process_network_output
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous': expanded_continuous
        }
        
        # Process network output - let the function handle sampling
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=False, 
            sample=False,
            random_continuous=False,  # This will trigger sampling inside process_network_output
            discrete_probs=discrete_one_hot,  # Pass the one-hot encoded discrete action
            observations=observation
        )
        
        # Store the original selected continuous mean for PPO calculations
        action_dict['discrete_action'] = discrete_action
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None

        discrete_distribution = torch.distributions.Categorical(logits=discrete_logits.squeeze(1))
        # override the logits part with that obtained from discrete_distribution (and getting the sampled action)
        action_dict['logits'] = discrete_distribution.log_prob(discrete_action_reshaped.squeeze(1))
        # action_dict['logits'] = discrete_logits.gather(-1, discrete_action.unsqueeze(-1)).squeeze(-1)

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs if process_state else None
        }

    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None, detach_continuous=False):
        raise NotImplementedError("check that normalization of the logits is correct")
        """Get logits, value, and entropy for PPO (only for discrete part)"""
        # Get discrete logits
        discrete_logits = self.policy.get_discrete_output(processed_observation)
        # apply categorical distribution to get the logits for the sampled action
        discrete_distribution = torch.distributions.Categorical(logits=discrete_logits.squeeze(1))
        
        # Need to reshape actions to match the logits dimension
        actions = discrete_action_indices.view(-1, 1, 1).expand(-1, discrete_logits.size(1), 1)
        
        intermediate = discrete_distribution.log_prob(discrete_action_indices).unsqueeze(1)
        # Gather logits for the specific actions that were taken
        # logits = discrete_logits.gather(-1, actions)
        # logits = discrete_distribution.log_prob(discrete_action_indices).gather(-1, actions).squeeze(-1)
        # logits = discrete_distribution.log_prob(discrete_action_indices).gather(-1, actions).squeeze(-1)
        
        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Calculate entropy (only for discrete part since continuous uses pathwise)
        entropy = self.get_entropy(discrete_logits)
        
        return intermediate, value, entropy
        # return logits, value, entropy

class GumbelSoftmaxAgent(HybridAgent):
    """
    Agent that uses Gumbel-Softmax relaxation for discrete actions to enable
    pathwise gradients through the entire network.
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # Get Gumbel-Softmax specific parameters
        agent_params = config['agent_params']
        self.initial_temperature = agent_params.get('initial_temperature', 1.0)
        self.min_temperature = agent_params.get('min_temperature', 0.1)
        self.temperature_decay = agent_params.get('temperature_decay', 0.995)
        self.use_straight_through = agent_params.get('use_straight_through', False)
        self.add_gumbel_noise = agent_params.get('add_gumbel_noise', True)
        
        # Current temperature (will be decayed during training)
        self.temperature = self.initial_temperature
        self.has_temperature = True
        
        super().__init__(config, feature_registry, device)
    
    def _get_required_losses(self):
        """GumbelSoftmaxAgent only needs pathwise and entropy losses."""
        return {
            'policy_gradient': False,  # No PPO needed
            'value': False,             # No value function
            'pathwise': True,          # Uses pathwise gradients
            'entropy': True            # No entropy loss, as actions are deterministic
        }
    
    def forward(self, observation, train=True):
        """Forward pass through the agent using Gumbel-Softmax for discrete actions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Apply Gumbel-Softmax to discrete logits during training
        if 'discrete' in raw_outputs:
            # Save pre-temperature logits
            logits = raw_outputs['discrete']
            raw_outputs['pre_temp_discrete_logits'] = logits.detach().clone()
            
            if train and self.add_gumbel_noise:
                # Sample from Gumbel(0, 1)
                uniform_samples = torch.rand_like(logits)
                gumbel_samples = -torch.log(-torch.log(uniform_samples + 1e-10) + 1e-10)
                
                # Add Gumbel noise to logits
                noisy_logits = logits + gumbel_samples
            else:
                # During inference or if noise is disabled, don't add noise
                noisy_logits = logits
            
            # Apply temperature scaling
            raw_outputs['discrete'] = noisy_logits / self.temperature
        
        # Process network output - let feature_registry handle argmax/softmax and straight-through
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=not train,  # Use argmax for inference, softmax for training
            sample=False,      # Never sample - we're using Gumbel noise instead
            straight_through=self.use_straight_through and train,  # Apply straight-through only during training
            observations=observation
        )
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices, continuous_samples=None, detach_continuous=False):
        """
        Get logits, value, and entropy for GumbelSoftmax agent.
        Returns None, None, and entropy (only from discrete head).
        """
        # Get policy outputs
        raw_outputs = self.policy(processed_observation, process_state=False)
        
        # Get entropy from discrete logits only
        entropy = self.get_entropy(raw_outputs['discrete'])
        
        return None, None, entropy

class FactoredGumbelSoftmaxAgent(GumbelSoftmaxAgent):
    """
    Factored version of GumbelSoftmaxAgent that:
    1. Uses separate networks for discrete and continuous outputs
    2. Evaluates continuous network for each possible discrete action
    3. Uses Gumbel-Softmax for differentiable discrete actions
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # Set flag to identify this as a factored agent
        self.factored = True
        super().__init__(config, feature_registry, device)
    
    def _init_policy(self, config):
        """Initialize policy with separate discrete and continuous networks"""
        return self._init_factored_policy(config)
    
    def forward(self, observation, train=True, process_state=True):
        """Forward pass: evaluate continuous network for all possible discrete actions"""
        processed_obs = self.feature_registry.prepare_inputs(observation)            
        # Get discrete logits
        discrete_logits = self.policy.get_discrete_output(processed_obs)

        processed_obs = torch.concat([processed_obs, discrete_logits.squeeze(1)], dim=-1)

        # Save pre-temperature logits
        raw_outputs = {'pre_temp_discrete_logits': discrete_logits.detach().clone()}
        
        # Apply Gumbel-Softmax during training
        if train and self.add_gumbel_noise:
            # Sample from Gumbel(0, 1)
            uniform_samples = torch.rand_like(discrete_logits)
            gumbel_samples = -torch.log(-torch.log(uniform_samples + 1e-10) + 1e-10)
            # Add Gumbel noise to logits
            noisy_logits = discrete_logits + gumbel_samples
        else:
            noisy_logits = discrete_logits
        
        # Apply temperature scaling
        scaled_logits = noisy_logits / self.temperature
        raw_outputs['discrete'] = scaled_logits
        
        # Create one-hot encodings for all possible discrete actions
        batch_size = processed_obs.size(0)
        n_discrete = discrete_logits.size(-1)
        
        # Expand observation to evaluate with each possible discrete action
        # [batch, features] -> [batch, n_discrete, features]
        expanded_obs = processed_obs.unsqueeze(1).expand(-1, n_discrete, -1)
        
        # Create one-hot encodings for all discrete actions
        one_hot = torch.eye(n_discrete, device=self.device)
        # Expand one-hot to match batch size
        # [n_discrete, n_discrete] -> [batch, n_discrete, n_discrete]
        one_hot = one_hot.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get continuous outputs for all discrete actions
        # Pass both expanded_obs and one_hot directly to get_continuous_output
        continuous_outputs = self.policy.get_continuous_output(expanded_obs, one_hot)
        continuous_outputs = continuous_outputs.squeeze(-1)

        # Store continuous outputs
        raw_outputs['continuous'] = continuous_outputs
        
        # Process network output
        action_dict = self.feature_registry.process_network_output(
            raw_outputs,
            argmax=not train,
            sample=False,
            straight_through=self.use_straight_through and train,
            observations=observation
        )
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs if process_state else None
        }   

class ContinuousOnlyAgent(BaseAgent):
    """
    Approach 2: Only using continuous actions, and approximating discontinuities 
    with predefined lines.
    - Only needs pathwise gradients
    - Only needs one continuous head in total
    - No value function needed
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # We don't need value network for this agent
        if 'nn_params' in config and 'value_network' in config['nn_params']:
            config['nn_params']['value_network']['enabled'] = False
        
        # Get existing parameters
        agent_params = config['agent_params']
        self.initial_temperature = agent_params.get('initial_temperature', 0.5)
        self.min_temperature = agent_params.get('min_temperature', 0.1)
        self.temperature_decay = agent_params.get('temperature_decay', 0.995)
        self.use_straight_through = agent_params.get('use_straight_through', False)
        
        # Add new parameter for zero-out indices
        self.zero_out_indices = agent_params.get('zero_out_action_dim', None)
        
        self.temperature = self.initial_temperature
        self.has_temperature = True
        
        super().__init__(config, feature_registry, device)
        self.feature_registry._initialize_sigmoid_scaling(device=self.device)
    
    def _init_policy(self, config):
        """Initialize the policy network with only continuous outputs"""
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # For this agent, we only need a single continuous output head
        # All actions will be represented as continuous values
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        
        # Override with a single continuous head and explicitly disable discrete head
        policy_params['policy_network']['heads'] = {
            'continuous': {
                'enabled': True,
                'size': 1,
                'activation': 'tanh'  # Use tanh to bound outputs between -1 and 1
            },
            'discrete': {
                'enabled': False,
                'size': 0
            }
        }
        
        # Use a continuous-only policy architecture
        return NeuralNetworkCreator().get_architecture(policy_params['policy_network']['name'])(policy_params, device=self.device)
    
    def forward(self, observation, train=True):
        """Forward pass through the agent using only continuous actions"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Process network output with zero-out functionality
        continuous_values = raw_outputs['continuous']
        action_dict = self.feature_registry.process_continuous_only_output(
            continuous_values, 
            temperature=self.temperature,
            argmax=not train,
            straight_through=self.use_straight_through and train,
            zero_out_indices=self.zero_out_indices,
            train=train
        )
        
        # Store the discrete probabilities in raw_outputs for logging
        if action_dict is not None:
            raw_outputs['discrete'] = action_dict['discrete_probs']  # Store discrete probabilities
            # You might also want to store other intermediate values
            raw_outputs['pre_temp_discrete_logits'] = continuous_values  # Store pre-temperature values
        
        return {
            'action_dict': action_dict,
            'value': None,  # No value network for this agent
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices, continuous_samples=None, detach_continuous=False):
        """Not used for this agent as we only have pathwise gradients"""
        # This method exists for API compatibility but will not be used for training
        raw_outputs = self.policy(processed_observation, process_state=False)
        return None, None, None

    def _get_required_losses(self):
        """ContinuousOnlyAgent only needs pathwise and entropy losses."""
        return {
            'policy_gradient': False,  # No PPO needed
            'value': False,            # No value function
            'pathwise': True,          # Uses pathwise gradients
            'entropy': False            # For exploration
        }

class GaussianPPOAgent(HybridAgent):
    """
    Approach 3: Use discrete actions and randomized continuous actions.
    - NN outputs parameters of a Gaussian distribution
    - Only needs PPO objective
    - Requires 1 head per discrete value, and 2 continuous heads per continuous value
      (mean and standard deviation of a gaussian)
    - Requires value network for PPO
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # Store parameters that we'll need after super().__init__
        self.fixed_std = config['agent_params'].get('fixed_std', False)
        self.include_continuous_entropy = config['agent_params'].get('include_continuous_entropy', False)
        self.n_continuous = feature_registry.get_network_dimensions()['n_continuous']
        self.device = device
        
        # Call parent's __init__ first
        super().__init__(config, feature_registry, device)
        
        # Now we can safely create the Parameter
        if self.fixed_std:
            self.log_std = nn.Parameter(torch.zeros(self.n_continuous, device=self.device))
    
    def _init_policy(self, config):
        """Initialize the policy network with Gaussian continuous outputs"""
        # Build policy sizes locally so fixed_std only outputs means
        network_dims = self.feature_registry.get_network_dimensions()
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        policy_params['policy_network']['heads']['discrete']['size'] = network_dims['n_discrete']

        if self.fixed_std:
            # Only output means when std is fixed
            continuous_size = network_dims['n_continuous']
        else:
            # Output mean and log_std for state-dependent std
            continuous_size = network_dims['n_continuous'] * 2

        policy_params['policy_network']['heads']['continuous']['size'] = continuous_size
        print(f"continuous size: {continuous_size}")

        return NeuralNetworkCreator().get_architecture(
            policy_params['policy_network']['name']
        )(policy_params, device=self.device)
    
    def forward(self, observation, train=True):
        """Forward pass through the agent"""
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
        # Split continuous outputs into mean and log_std
        continuous_outputs = raw_outputs['continuous']
        if self.fixed_std:
            continuous_mean = continuous_outputs
            continuous_log_std = self.log_std.expand_as(continuous_mean)
        else:
            n_continuous = continuous_outputs.size(-1) // 2
            # First half is always mean
            continuous_mean = continuous_outputs[..., :n_continuous]
            # Second half is log_std
            continuous_log_std = continuous_outputs[..., n_continuous:]
        
        # Store these in raw_outputs
        raw_outputs['continuous_mean'] = continuous_mean
        raw_outputs['continuous_log_std'] = continuous_log_std

        apply_scaling = not self.policy_config.get('disable_continuous_scaling', False)
        
        # Process network output
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=not train, 
            sample=train,
            random_continuous=train,
            observations=observation,
            apply_scaling=apply_scaling
        )
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        # Store the raw continuous samples for later use in PPO
        if 'continuous_samples' in action_dict:
            action_dict['raw_continuous_samples'] = action_dict['continuous_samples']

        if train:
            # Align training-time outputs with FactoredGaussianPPOAgent
            discrete_action_indices = action_dict.get('discrete_action_indices')
            if discrete_action_indices is not None:
                action_dict['discrete_action'] = discrete_action_indices

            continuous_mean_reshaped = self.feature_registry.reshape_continuous_output(continuous_mean)
            action_dict['selected_continuous_mean'] = continuous_mean_reshaped

            continuous_samples = action_dict.get('raw_continuous_samples')
            continuous_log_std_reshaped = self.feature_registry.reshape_continuous_output(continuous_log_std)

            # Compute log probabilities for PPO, matching FactoredGaussianPPOAgent behavior
            if (discrete_action_indices is not None and continuous_samples is not None
                    and continuous_mean_reshaped is not None and continuous_log_std_reshaped is not None):
                discrete_logits = raw_outputs.get('discrete')
                clamped_log_std = torch.clamp(continuous_log_std_reshaped, min=-20, max=2)
                continuous_std = torch.exp(clamped_log_std)

                continuous_samples = self.feature_registry.reshape_continuous_output(continuous_samples)
                continuous_log_probs = self.get_continuous_log_probs(
                    continuous_mean_reshaped,
                    continuous_std,
                    continuous_samples,
                    discrete_action_indices
                )
                if continuous_log_probs.dim() > 1:
                    continuous_log_probs = continuous_log_probs.sum(dim=-1, keepdim=True)

                if discrete_logits is not None:
                    discrete_log_probs = self.get_discrete_log_probs(discrete_logits, discrete_action_indices)
                    if discrete_log_probs.dim() > 1:
                        discrete_log_probs = discrete_log_probs.sum(dim=-1, keepdim=True)
                    action_dict['log_probs'] = discrete_log_probs + continuous_log_probs
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs
        }
    
    def get_entropy(self, logits):
        """Get entropy of a categorical distribution"""
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.entropy()
    
    def get_gaussian_entropy(self, log_std):
        """Get entropy of a Gaussian distribution
        This gradient is slightly biased when log_std is a vector, since we don't consider the cross-term coming from
        log-prob of discrete actions and the entropy of the continuous actions. Still, it is a good approximation
        that aids in exploration.
        
        Args:
            log_std: Log standard deviation. Can be scalar or vector.
                    For fixed std: scalar tensor
                    For state-dependent std: vector tensor with shape matching the mean
        """
        # Entropy of a Gaussian is 0.5 * log(2*pi*e*sigma^2)
        # = 0.5 + 0.5*log(2*pi) + log_std
        return 0.5 + 0.5 * torch.log(2 * torch.tensor(torch.pi, device=log_std.device)) + log_std
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None, detach_continuous=False):
        """
        Get combined log probabilities (discrete + continuous), value, and entropy for PPO.
        This is the non-factored version: the policy outputs both discrete logits and
        a Gaussian distribution over continuous values for all sub-ranges.
        """
        # Get policy outputs
        raw_outputs = self.policy(processed_observation, process_state=False)
        discrete_logits = raw_outputs['discrete']

        # Get continuous mean/log_std from the policy outputs
        continuous_outputs = raw_outputs.get('continuous')
        continuous_mean = None
        continuous_log_std = None
        if continuous_outputs is not None:
            if self.fixed_std:
                continuous_mean = continuous_outputs
                continuous_log_std = self.log_std.expand_as(continuous_mean)
            else:
                n_continuous = continuous_outputs.size(-1) // 2
                continuous_mean = continuous_outputs[..., :n_continuous]
                continuous_log_std = continuous_outputs[..., n_continuous:]

            # Reshape to [batch, n_stores, n_sub_ranges] for consistency
            continuous_mean = self.feature_registry.reshape_continuous_output(continuous_mean)
            continuous_log_std = self.feature_registry.reshape_continuous_output(continuous_log_std)

        # Compute discrete log probabilities
        discrete_log_probs = self.get_discrete_log_probs(discrete_logits, discrete_action_indices)
        if discrete_log_probs.dim() > 1:
            discrete_log_probs = discrete_log_probs.sum(dim=-1, keepdim=True)

        # Compute continuous log probabilities (only if samples provided)
        continuous_log_probs = None
        if continuous_samples is not None and continuous_mean is not None and continuous_log_std is not None:
            if detach_continuous:
                continuous_mean = continuous_mean.detach()
                continuous_log_std = continuous_log_std.detach()
            continuous_log_std = torch.clamp(continuous_log_std, min=-20, max=2)
            continuous_std = torch.exp(continuous_log_std)

            # Ensure continuous_samples are in [batch, n_stores, n_sub_ranges] shape
            continuous_samples = self.feature_registry.reshape_continuous_output(continuous_samples)

            continuous_log_probs = self.get_continuous_log_probs(
                continuous_mean,
                continuous_std,
                continuous_samples,
                discrete_action_indices
            )
            if continuous_log_probs.dim() > 1:
                continuous_log_probs = continuous_log_probs.sum(dim=-1, keepdim=True)

        total_log_probs = (
            discrete_log_probs if continuous_log_probs is None
            else (discrete_log_probs + continuous_log_probs)
        )

        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None

        # Entropy: discrete, optionally include continuous (Gaussian)
        discrete_entropy = self.get_entropy(discrete_logits)
        if discrete_entropy.dim() > 1:
            discrete_entropy = discrete_entropy.sum(dim=-1)

        continuous_entropy = 0
        if self.include_continuous_entropy and continuous_log_std is not None:
            continuous_entropy = self.get_gaussian_entropy(continuous_log_std)
            if continuous_entropy.dim() > 1:
                continuous_entropy = continuous_entropy.sum(dim=-1)
            if continuous_entropy.dim() > 1:
                continuous_entropy = continuous_entropy.sum(dim=-1)

        total_entropy = discrete_entropy + continuous_entropy

        return total_log_probs, value, total_entropy

    def _get_required_losses(self):
        """GaussianPPOAgent needs policy gradient, value, and entropy losses."""
        return {
            'policy_gradient': True,   # For both discrete and continuous actions
            'value': True,             # For PPO advantage estimation
            'pathwise': False,         # No pathwise gradients
            'entropy': True            # For exploration
        }

    def parameters(self):
        """Return parameters for optimization, including fixed log_std if used"""
        params = super().parameters()
        if self.fixed_std:
            # Convert params to list if it's not already
            params = list(params)
            # Add the fixed log_std parameter
            params.append(self.log_std)
        return params
    
class FactoredGaussianPPOAgent(GaussianPPOAgent):
    """
    Agent that uses a factored approach to hybrid actions:
    1. First pass: discrete action distribution from state
    2. Second pass: continuous action mean from state + sampled discrete action
    
    Uses a single continuous head and a single scalar parameter for standard deviation.
    """
    def __init__(self, config, feature_registry=None, device='cpu'):
        # Store dimensions before calling super().__init__
        self.device = device
        if feature_registry:
            self.n_discrete = feature_registry.get_network_dimensions()['n_discrete']
            self.n_continuous = feature_registry.get_network_dimensions()['n_continuous']
            self.n_stores = feature_registry.n_stores
            self.n_sub_ranges = feature_registry.n_sub_ranges
        
        # Set flag to identify this as a factored agent
        self.factored = True
        
        # Check if we're using fixed or state-dependent std
        self.fixed_std = config.get('agent_params', {}).get('fixed_std', True)
        
        # Initialize base class
        super().__init__(config, feature_registry, device)
        
        # Handle log_std parameter based on fixed_std setting
        if hasattr(self, 'log_std'):
            # Delete existing parameter if it exists
            del self.log_std
            
        if self.fixed_std:
            # Create new scalar parameter for fixed std
            self.log_std = nn.Parameter(torch.zeros(1, device=device))
        else:
            # No log_std parameter needed - it will come from network output
            self.log_std = None
    
    def _init_policy(self, config):
        """Initialize policy with separate discrete and continuous networks"""
        return self._init_factored_policy(config)  # Uses default continuous_size=1
    
    def forward(self, observation, train=True, process_state=True):
        """Forward pass: first sample discrete action, then get continuous mean"""
        # Process observation to get features if needed
        if process_state:
            processed_obs = self.feature_registry.prepare_inputs(observation)
        else:
            processed_obs = observation
            
        # First get discrete distribution
        discrete_logits = self.policy.get_discrete_output(processed_obs)
        
        # Choose discrete action based on train mode
        if train:
            discrete_distribution = torch.distributions.Categorical(logits=discrete_logits.squeeze(1))
            discrete_action = discrete_distribution.sample()
        else:
            discrete_action = discrete_logits.squeeze(1).argmax(dim=-1)

        
        discrete_action_reshaped = discrete_action.unsqueeze(-1)
        # Create one-hot encoding for the sampled discrete action
        discrete_one_hot = torch.zeros_like(discrete_logits.squeeze(1)).scatter_(-1, discrete_action_reshaped, 1)
        # Keep a single store dimension; this will broadcast across stores downstream
        discrete_one_hot_stores = discrete_one_hot.unsqueeze(1)
        
        # Get continuous output (conditioned on sampled discrete action)
        continuous_output = self.policy.get_continuous_output(
            processed_obs,
            discrete_one_hot,
            include_std=not self.fixed_std
        )
        
        if self.fixed_std:
            # Fixed std: continuous_output is just the mean per store
            selected_continuous_mean = continuous_output.squeeze(1)
            # Use the scalar log_std parameter (shared across products)
            selected_log_std = self.log_std
        else:
            # State-dependent std: continuous_output contains mean per store + scalar log_std
            continuous_output = continuous_output.squeeze(1)
            n_continuous = self.n_stores
            selected_continuous_mean = continuous_output[..., :n_continuous]
            selected_log_std = continuous_output[..., n_continuous:]
        
        # Expand mean/log_std to [batch, n_stores, n_sub_ranges]
        mean_per_store = selected_continuous_mean
        mean_per_store_sub = mean_per_store.unsqueeze(-1).expand(
            -1, self.n_stores, self.n_sub_ranges
        )
        if self.fixed_std:
            expanded_log_std = selected_log_std.expand_as(mean_per_store_sub)
        else:
            expanded_log_std = selected_log_std.expand_as(mean_per_store).unsqueeze(-1).expand(
                -1, self.n_stores, self.n_sub_ranges
            )
        
        # Create raw_outputs dict in the format expected by process_network_output
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous_mean': mean_per_store_sub,
            'continuous_log_std': expanded_log_std,
            'continuous': mean_per_store_sub
        }
        
        # Process network output - let the function handle sampling
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=False, # don't use argmax for discrete
            sample=False, # don't sample for discrete
            random_continuous=train,  # Sample for training, don't sample for inference (use mean as the continuous action)
            discrete_probs=discrete_one_hot_stores,  # Pass per-store one-hot for shape consistency
            observations=observation
        )
        
        if train:
            # Store the original selected continuous mean for PPO calculations
            action_dict['selected_continuous_mean'] = mean_per_store
            action_dict['discrete_action'] = discrete_action
            
            # Calculate and store log probabilities for PPO using BaseAgent helper function
            # Use the same clamped log_std that was used for sampling to ensure consistency
            if self.fixed_std:
                clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
                continuous_std = torch.exp(clamped_log_std).expand_as(mean_per_store_sub)
            else:
                # For state-dependent std, use the selected log_std directly
                clamped_log_std = torch.clamp(selected_log_std, min=-20, max=2)
                continuous_std = torch.exp(clamped_log_std).expand_as(mean_per_store).unsqueeze(-1).expand_as(mean_per_store_sub)
            
            mean_per_store = mean_per_store_sub
            std_per_store = continuous_std
            samples_per_store = action_dict['raw_continuous_samples']
            
            discrete_log_probs = self.get_discrete_log_probs(discrete_logits, discrete_action)
            if discrete_log_probs.dim() > 1:
                discrete_log_probs = discrete_log_probs.sum(dim=-1, keepdim=True)
            continuous_log_probs = self.get_continuous_log_probs(
                mean_per_store,
                std_per_store,
                samples_per_store,
                discrete_action
            )
            # Sum log probs across stores
            if continuous_log_probs.dim() > 1:
                continuous_log_probs = continuous_log_probs.sum(dim=-1, keepdim=True)
            action_dict['log_probs'] = discrete_log_probs + continuous_log_probs
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': processed_obs if process_state else None
        }
    
    
    def parameters(self):
        """Return all trainable parameters"""
        params = list(self.policy.parameters())
        if self.value_net is not None:
            params += list(self.value_net.parameters())
        if self.log_std is not None:
            params.append(self.log_std)
        return params
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None, detach_continuous=False):
        """
        Get combined log probabilities (discrete + continuous), value, and entropy for PPO
        
        Args:
            processed_observation: Processed observation tensor
            discrete_action_indices: Indices of discrete actions taken during sampling [batch]
            continuous_samples: Continuous action samples (used for calculating log probabilities)
            
        Returns:
            total_log_probs: Combined log probabilities of discrete and continuous actions
            value: Value estimate
            total_entropy: Total entropy
        """
        # Get discrete logits from policy
        discrete_logits = self.policy.get_discrete_output(processed_observation)  # [batch, n_discrete]
        
        # Create one-hot encoding for discrete actions using BaseAgent helper function
        discrete_one_hot = self.get_discrete_one_hot(discrete_logits, discrete_action_indices)
        if discrete_one_hot.dim() == 3:
            discrete_one_hot = discrete_one_hot.squeeze(1)
        
        # Get continuous output conditioned on the discrete action
        continuous_output = self.policy.get_continuous_output(
            processed_observation, 
            discrete_one_hot,
            include_std=not self.fixed_std
        )
        
        if self.fixed_std:
            # Fixed std: continuous_output is just the mean per store
            selected_continuous_mean = continuous_output.squeeze(1)
            # Use the scalar log_std parameter
            clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
            continuous_std = torch.exp(clamped_log_std).expand_as(selected_continuous_mean)
            log_std_expanded = clamped_log_std.expand_as(selected_continuous_mean)
        else:
            # State-dependent std: continuous_output contains mean per store + scalar log_std
            continuous_output = continuous_output.squeeze(1)
            n_continuous = self.n_stores
            selected_continuous_mean = continuous_output[..., :n_continuous]
            selected_log_std = continuous_output[..., n_continuous:]
            clamped_log_std = torch.clamp(selected_log_std, min=-20, max=2)
            continuous_std = torch.exp(clamped_log_std).expand_as(selected_continuous_mean)
            log_std_expanded = clamped_log_std.expand_as(selected_continuous_mean)

        if detach_continuous:
            selected_continuous_mean = selected_continuous_mean.detach()
            continuous_std = continuous_std.detach()
            log_std_expanded = log_std_expanded.detach()
        
        # Use the helper function to calculate total log probabilities
        mean_per_store = selected_continuous_mean.unsqueeze(-1).expand(
            -1, self.n_stores, self.n_sub_ranges
        )
        std_per_store = continuous_std.unsqueeze(-1).expand(
            -1, self.n_stores, self.n_sub_ranges
        )
        continuous_log_probs = None
        if continuous_samples is not None:
            continuous_log_probs = self.get_continuous_log_probs(
                mean_per_store,
                std_per_store,
                continuous_samples,
                discrete_action_indices
            )
            # Sum log probs across stores
            if continuous_log_probs.dim() > 1:
                continuous_log_probs = continuous_log_probs.sum(dim=-1, keepdim=True)
        discrete_log_probs = self.get_discrete_log_probs(discrete_logits, discrete_action_indices)
        if discrete_log_probs.dim() > 1:
            discrete_log_probs = discrete_log_probs.sum(dim=-1, keepdim=True)
        total_log_probs = discrete_log_probs if continuous_log_probs is None else (discrete_log_probs + continuous_log_probs)
        
        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Calculate total entropy (optionally include continuous dims)
        discrete_entropy = self.get_entropy(discrete_logits)
        if discrete_entropy.dim() > 1:
            discrete_entropy = discrete_entropy.sum(dim=-1)
        continuous_entropy = 0
        if self.include_continuous_entropy:
            log_std_per_store = log_std_expanded.unsqueeze(-1).expand(
                -1, self.n_stores, self.n_sub_ranges
            )
            continuous_entropy = self.get_gaussian_entropy(log_std_per_store)
            if continuous_entropy.dim() > 1:
                continuous_entropy = continuous_entropy.sum(dim=-1)
            if continuous_entropy.dim() > 1:
                continuous_entropy = continuous_entropy.sum(dim=-1)
        total_entropy = discrete_entropy + continuous_entropy
        
        return total_log_probs, value, total_entropy
