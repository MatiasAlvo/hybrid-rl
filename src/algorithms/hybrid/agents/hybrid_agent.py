import torch
import torch.nn.functional as F
from src.algorithms.common.policies.policy import HybridPolicy, NeuralNetworkCreator
# from src.algorithms.common.policies.policy import HybridPolicy, NeuralNetworkCreator, ContinuousPolicy
from src.algorithms.common.values.value_network import ValueNetwork
import torch.nn as nn
import torch.distributions as dist
import numpy as np

class BaseAgent(nn.Module):
    def __init__(self, config, feature_registry=None, device='cpu'):
        super().__init__()
        self.device = device
        self.feature_registry = feature_registry
        
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
        if hasattr(value_params, 'observation_keys'):
            value_net.observation_keys = value_params['observation_keys']
        elif hasattr(self.feature_registry, 'get_observation_keys'):
            value_net.observation_keys = self.feature_registry.get_observation_keys()
        else:
            # Default fallback - you may need to adjust this based on your implementation
            value_net.observation_keys = ['store_inventories']
            
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
    
    def get_logits_value_and_entropy(self, processed_observation, action_indices):
        """Get logits for specific actions, value, and entropy. Override in subclasses."""
        raise NotImplementedError
    
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
        
        return NeuralNetworkCreator().get_architecture(policy_params['policy_network']['name'])(policy_params, device=self.device)

    def forward(self, observation, train=True):
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
    
    def get_logits_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
        """
        Get logits, value, and entropy for PPO
        
        Args:
            processed_observation: Processed observation tensor
            discrete_action_indices: Indices of discrete actions taken during sampling
            continuous_samples: Continuous action samples (ignored in base HybridAgent)
            
        Returns:
            logits: Log probabilities of the discrete actions
            value: Value from the value network
            entropy: Entropy of the policy
        """
        # Get policy outputs
        raw_outputs = self.policy(processed_observation, process_state=False)
        
        # Need to reshape actions to match the logits dimension
        actions = discrete_action_indices.view(-1, 1, 1).expand(-1, raw_outputs['discrete'].size(1), 1)
        
        # Gather logits for the specific actions that were taken
        logits = raw_outputs['discrete'].gather(-1, actions).squeeze(-1)
        
        value = self.value_net(processed_observation.detach(), process_state=False) if self.value_net else None
        entropy = self.get_entropy(raw_outputs['discrete'])
        
        return logits, value, entropy

    def _get_required_losses(self):
        """HybridAgent needs all loss components."""
        return {
            'policy_gradient': True,   # For discrete actions
            'value': True,             # For PPO advantage estimation
            'pathwise': True,          # For continuous actions
            'entropy': True            # For exploration
        }


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
            'entropy': False            # No entropy loss, as actions are deterministic
        }
    
    def forward(self, observation, train=True):
        """Forward pass through the agent using Gumbel-Softmax for discrete actions"""
        # Get raw outputs from policy
        raw_outputs = self.policy(observation)
        
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
            straight_through=self.use_straight_through and train  # Apply straight-through only during training
        )
        
        # Get value if value network exists
        value = self.value_net(observation) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs
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
        self.zero_out_indices = agent_params.get('zero_out_action_dim', [])
        
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
        # Get raw outputs from policy
        raw_outputs = self.policy(observation)
        
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
            'raw_outputs': raw_outputs
        }
    
    def get_logits_value_and_entropy(self, processed_observation, action_indices):
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
        self.n_continuous = feature_registry.get_network_dimensions()['n_continuous']
        self.device = device
        
        # Call parent's __init__ first
        super().__init__(config, feature_registry, device)
        
        # Now we can safely create the Parameter
        if self.fixed_std:
            self.log_std = nn.Parameter(torch.zeros(self.n_continuous, device=self.device))
    
    def _init_policy(self, config):
        """Initialize the policy network with Gaussian continuous outputs"""
        # Call parent's _init_policy with random_continuous=True
        return super()._init_policy(config, random_continuous=True)
    
    def forward(self, observation, train=True):
        """Forward pass through the agent"""
        # Get raw outputs from policy
        raw_outputs = self.policy(observation)
        
        # Split continuous outputs into mean and log_std
        continuous_outputs = raw_outputs['continuous']
        n_continuous = continuous_outputs.size(-1) // 2
        
        # First half is always mean
        continuous_mean = continuous_outputs[..., :n_continuous]
        
        # For log_std, either use the state-dependent output or the fixed parameter
        if self.fixed_std:
            continuous_log_std = self.log_std.expand_as(continuous_mean)
        else:
            continuous_log_std = continuous_outputs[..., n_continuous:]
        
        # Store these in raw_outputs
        raw_outputs['continuous_mean'] = continuous_mean
        raw_outputs['continuous_log_std'] = continuous_log_std
        
        # Process network output
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=False, 
            sample=True,
            random_continuous=True
        )
        
        # Get value if value network exists
        value = self.value_net(observation) if self.value_net is not None else None
        
        # Store the raw continuous samples for later use in PPO
        if 'continuous_samples' in action_dict:
            action_dict['raw_continuous_samples'] = action_dict['continuous_samples']
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs
        }
    
    def get_entropy(self, logits):
        """Get entropy of a categorical distribution"""
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.entropy()
    
    def get_gaussian_entropy(self, log_std):
        """Get entropy of a Gaussian distribution"""
        # Entropy of a Gaussian is 0.5 * log(2*pi*e*sigma^2)
        # = 0.5 + 0.5*log(2*pi) + log_std
        return 0.5 + 0.5 * torch.log(2 * torch.tensor(torch.pi, device=log_std.device)) + log_std
    
    def get_logits_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
        """
        Get combined log probabilities (discrete + continuous), value, and entropy for PPO
        
        Args:
            processed_observation: Processed observation tensor
            discrete_action_indices: Indices of discrete actions taken during sampling
            continuous_samples: Continuous action samples (used for calculating log probabilities)
            
        Returns:
            total_logits: Combined log probabilities of discrete and continuous actions
            value: Value estimate
            total_entropy: Total entropy
        """
        # Get policy outputs
        raw_outputs = self.policy(processed_observation, process_state=False)
        
        # Calculate discrete logits
        discrete_logits = raw_outputs['discrete']
        
        # Need to reshape actions to match the logits dimension
        actions = discrete_action_indices.view(-1, 1, 1).expand(-1, discrete_logits.size(1), 1)
        
        # Gather logits for the specific actions that were taken
        discrete_logprobs = discrete_logits.gather(-1, actions).squeeze(-1)
        
        # Initialize total log probability with discrete logits
        total_logits = discrete_logprobs
        
        # Handle continuous outputs
        if 'continuous' in raw_outputs:
            continuous_outputs = raw_outputs['continuous']
            n_continuous = continuous_outputs.size(-1) // 2
            
            # Split continuous outputs into means and stds
            continuous_mean = continuous_outputs[..., :n_continuous]
            
            if self.fixed_std:
                continuous_log_std = self.log_std.expand_as(continuous_mean)
            else:
                continuous_log_std = continuous_outputs[..., n_continuous:]
            
            continuous_std = torch.exp(continuous_log_std)
            normal_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            
            if continuous_samples is not None:
                continuous_log_probs = normal_dist.log_prob(continuous_samples)
                selected_continuous_log_probs = continuous_log_probs.gather(-1, actions).squeeze(-1)
                total_logits = total_logits + selected_continuous_log_probs
                # total_logits += selected_continuous_log_probs
        
        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Calculate total entropy
        discrete_entropy = self.get_entropy(discrete_logits)
        continuous_entropy = normal_dist.entropy().sum(-1) if 'continuous' in raw_outputs else 0
        total_entropy = discrete_entropy + continuous_entropy
        
        return total_logits, value, total_entropy

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
        
        # Set flag to identify this as a factored agent
        self.factored = True
        
        # Initialize base class
        super().__init__(config, feature_registry, device)
        
        # Replace vector log_std with scalar
        if hasattr(self, 'log_std'):
            # Delete existing parameter if it exists
            del self.log_std
        # Create new scalar parameter
        self.log_std = nn.Parameter(torch.zeros(1, device=device))
    
    def _init_policy(self, config):
        """Initialize policy with separate discrete and continuous networks"""
        # Get network dimensions from feature registry
        network_dims = self.feature_registry.get_network_dimensions()
        
        # Update config
        policy_params = config['nn_params']
        policy_params['policy_network']['input_size'] = network_dims['input_size']
        policy_params['policy_network']['heads']['discrete']['size'] = network_dims['n_discrete']
        
        # For continuous head, we only output a single value (not per discrete action)
        # This will be expanded to the right shape later
        policy_params['policy_network']['heads']['continuous']['size'] = 1
        
        # Create new policy network from the FactoredPolicy class defined in policy.py
        policy_class = NeuralNetworkCreator().get_architecture("factored_policy")
        return policy_class(policy_params, device=self.device)
    
    def forward(self, observation, train=True, process_state=True):
        """Forward pass: first sample discrete action, then get continuous mean"""
        # Process observation to get features if needed
        if process_state:
            processed_obs = observation['store_inventories'].flatten(start_dim=1)
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
        
        # Get continuous mean (conditioned on sampled discrete action)
        selected_continuous_mean = self.policy.get_continuous_output(processed_obs, discrete_one_hot.squeeze(1))
        
        # Expand continuous mean to match discrete action dimension
        expanded_continuous_mean = selected_continuous_mean.unsqueeze(1).expand(-1, discrete_logits.size(1), discrete_logits.size(2))
        
        # Use the scalar log_std parameter expanded to match the expanded mean's shape
        expanded_log_std = self.log_std.expand_as(expanded_continuous_mean)
        
        # Create raw_outputs dict in the format expected by process_network_output
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous_mean': expanded_continuous_mean,
            'continuous_log_std': expanded_log_std,
            'continuous': expanded_continuous_mean
        }
        
        # Process network output - let the function handle sampling
        action_dict = self.feature_registry.process_network_output(
            raw_outputs, 
            argmax=False, 
            sample=False,
            random_continuous=True,  # This will trigger sampling inside process_network_output
            discrete_probs=discrete_one_hot  # Pass the one-hot encoded discrete action
        )
        
        # Store the original selected continuous mean for PPO calculations
        action_dict['selected_continuous_mean'] = selected_continuous_mean
        action_dict['discrete_action'] = discrete_action
        
        # Get value if value network exists
        value = self.value_net(observation) if self.value_net is not None else None

        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs
        }
    
    
    def parameters(self):
        """Return all trainable parameters"""
        params = list(self.policy.parameters())
        if self.value_net is not None:
            params += list(self.value_net.parameters())
        params.append(self.log_std)
        return params
    
    def get_logits_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
        """
        Get combined log probabilities (discrete + continuous), value, and entropy for PPO
        
        Args:
            processed_observation: Processed observation tensor
            discrete_action_indices: Indices of discrete actions taken during sampling [batch]
            continuous_samples: Continuous action samples (used for calculating log probabilities)
            
        Returns:
            total_logits: Combined log probabilities of discrete and continuous actions
            value: Value estimate
            total_entropy: Total entropy
        """
        # Get policy outputs
        discrete_logits = self.policy.get_discrete_output(processed_observation)  # [batch, n_discrete]
        
        # Need to reshape actions to match the logits dimension
        # discrete_action_indices is [batch], we need [batch, 1] for gather
        actions = discrete_action_indices.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        
        # Gather logits for the specific actions that were taken
        discrete_logprobs = discrete_logits.gather(-1, actions)  # [batch, 1, 1]
        discrete_logprobs = discrete_logprobs.squeeze(-1).squeeze(-1)  # [batch]
        
        # Initialize total log probability with discrete logits
        total_logits = discrete_logprobs
        
        # Handle continuous outputs
        # Create one-hot encoding for discrete actions
        discrete_one_hot = torch.zeros_like(discrete_logits)
        discrete_one_hot.scatter_(-1, actions, 1)
        discrete_one_hot = discrete_one_hot.squeeze(1)
        
        # Get continuous mean conditioned on the discrete action
        selected_continuous_mean = self.policy.get_continuous_output(
            processed_observation, 
            discrete_one_hot  # Already in correct shape
        )
        
        # Calculate continuous log probabilities
        if continuous_samples is not None:
            continuous_std = torch.exp(self.log_std)
            normal_dist = torch.distributions.Normal(selected_continuous_mean[:, 0], continuous_std.expand_as(selected_continuous_mean[:, 0]))
            selected_continuous_samples = continuous_samples.gather(-1, actions)  # [batch, 1, 1]
            selected_continuous_samples = selected_continuous_samples.squeeze(-1).squeeze(-1)  # [batch]
            continuous_log_probs = normal_dist.log_prob(selected_continuous_samples)
            total_logits = total_logits + continuous_log_probs
        
        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Calculate total entropy
        discrete_entropy = self.get_entropy(discrete_logits)
        continuous_entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std  # Gaussian entropy formula
        total_entropy = discrete_entropy + continuous_entropy.sum()
        
        return total_logits.unsqueeze(-1), value, total_entropy

    def get_logits_value_and_entropy_old(self, processed_observation, discrete_action_indices, continuous_samples=None):
        """
        Calculate log probabilities, value estimates, and entropy for PPO updates
        
        Args:
            processed_observation: Processed observation tensor
            discrete_action_indices: Indices of discrete actions taken during sampling
            continuous_samples: Continuous action samples (used for calculating log probabilities)
                Shape: [batch, 1, n_features]
            
        Returns:
            total_logits: Combined log probabilities of discrete and continuous actions
            value: Value estimate
            total_entropy: Total entropy
        """
        # Step 1: Get discrete logits from policy
        discrete_logits = self.policy.get_discrete_output(processed_observation)
        
        # Step 2: Calculate discrete log probabilities
        # Apply softmax to get probabilities, then take log
        # discrete_log_probs = F.log_softmax(discrete_logits, dim=-1)
        discrete_log_probs = discrete_logits.gather(-1, discrete_action_indices).squeeze(-1)
        
        # Step 3: Get log probs for selected discrete actions
        # Reshape action indices to [batch, 1, 1] to match gather dimension
        discrete_action_reshaped = discrete_action_indices.view(-1, 1, 1)
        selected_discrete_log_probs = discrete_log_probs.gather(-1, discrete_action_reshaped).squeeze(-1).squeeze(1)
        
        # Step 4: Create one-hot encoding for selected discrete actions
        discrete_one_hot = torch.zeros_like(discrete_logits)
        discrete_one_hot.scatter_(-1, discrete_action_reshaped, 1)
        
        # Step 5: Get continuous mean conditioned on the discrete action
        # Squeeze out the middle dimension to match expected input shape
        selected_continuous_mean = self.policy.get_continuous_output(
            processed_observation, 
            discrete_one_hot.squeeze(1)
        )
        
        # Step 6: Calculate log probabilities for continuous actions if provided
        total_logits = selected_discrete_log_probs  # Start with discrete log probs
        
        if continuous_samples is not None:
            # Get standard deviation from our scalar parameter
            continuous_log_std = self.log_std.expand_as(selected_continuous_mean)
            continuous_std = torch.exp(continuous_log_std)
            
            # Create normal distribution
            normal_dist = torch.distributions.Normal(selected_continuous_mean.squeeze(1), continuous_std.squeeze(1))
            
            # Simplify by using squeeze directly on continuous_samples
            # This assumes continuous_samples has shape [batch, 1, n_features]
            # After squeeze(1), it becomes [batch, n_features] which matches selected_continuous_mean
            selected_continuous_samples = continuous_samples.gather(-1, discrete_action_reshaped).squeeze(-1).squeeze(1)
            
            # Calculate log probabilities for the samples
            continuous_log_probs = normal_dist.log_prob(selected_continuous_samples)
            
            # Add continuous log probs to total
            total_logits = total_logits + continuous_log_probs
        
        # Step 7: Calculate value estimate
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Step 8: Calculate entropy
        # For discrete part - need to remove extra dimension for entropy calculation
        discrete_entropy = torch.distributions.Categorical(logits=discrete_logits.squeeze(1)).entropy()
        
        # For continuous part
        continuous_entropy = torch.distributions.Normal(
            selected_continuous_mean, continuous_std
        ).entropy().sum(dim=-1)
        
        # Total entropy
        total_entropy = discrete_entropy + continuous_entropy
        
        return total_logits, value, total_entropy