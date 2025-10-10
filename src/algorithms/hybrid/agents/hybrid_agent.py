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
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices):
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
        """
        Compute log probabilities for continuous actions using Normal distribution
        
        Args:
            continuous_mean: Mean of continuous actions [batch, n_continuous] or [batch, n_stores, n_continuous]
            continuous_std: Standard deviation of continuous actions [batch, n_continuous] or [batch, n_stores, n_continuous]
            continuous_samples: Sampled continuous actions [batch, n_continuous] or [batch, n_stores, n_continuous]
            discrete_action_indices: Optional discrete action indices for factored approaches [batch] or [batch, n_stores]
            
        Returns:
            continuous_log_probs: Log probabilities of the sampled continuous actions
        """
        # Create normal distribution
        normal_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        
        # Calculate log probabilities for the samples
        continuous_log_probs = normal_dist.log_prob(continuous_samples)
        
        # Sum over continuous dimensions if there are multiple
        if continuous_log_probs.dim() > 1 and continuous_log_probs.size(-1) > 1:
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
                # Fixed std: only output mean (size = 1)
                policy_params['policy_network']['heads']['continuous']['size'] = 1
            else:
                # State-dependent std: output mean + log_std (size = 2)
                policy_params['policy_network']['heads']['continuous']['size'] = 2
        
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
        if isinstance(raw_outputs, dict) and 'discrete' in raw_outputs:
            discrete_logits = raw_outputs['discrete']
            if torch.isnan(discrete_logits).any() or torch.isinf(discrete_logits).any():
                print("Warning: NaN or Inf in discrete logits from policy")
                print("Discrete logits stats:", 
                      f"range [{discrete_logits.min().item():.3f}, {discrete_logits.max().item():.3f}], "
                      f"mean {discrete_logits.mean().item():.3f}")
        
        # Process discrete outputs
        discrete_output = self.feature_registry.process_discrete_output(
            raw_outputs['discrete'],
            # argmax=False,  # Use argmax for inference, sample for training
            argmax=not train,  # Use argmax for inference, sample for training
            sample=True,      # Sample during training
            straight_through=False
        )
        
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
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
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
        
        # Get continuous mean (conditioned on sampled discrete action)
        selected_continuous = self.policy.get_continuous_output(processed_obs, discrete_one_hot.squeeze(1))
        
        # First squeeze out the extra dimension
        selected_continuous = selected_continuous.squeeze(2)  # Remove the extra dimension
        expanded_continuous = selected_continuous.unsqueeze(1).expand(-1, discrete_logits.size(1), discrete_logits.size(2))
        
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

    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
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
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices, continuous_samples=None):
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
    
    def get_log_probs_value_and_entropy(self, processed_observation, action_indices):
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
        # Prepare inputs using feature registry (normalizes and flattens)
        processed_obs = self.feature_registry.prepare_inputs(observation)
        
        # Get raw outputs from policy
        raw_outputs = self.policy(processed_obs, process_state=False)
        
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
            random_continuous=True,
            observations=observation
        )
        
        # Get value if value network exists
        value = self.value_net(processed_obs, process_state=False) if self.value_net is not None else None
        
        # Store the raw continuous samples for later use in PPO
        if 'continuous_samples' in action_dict:
            action_dict['raw_continuous_samples'] = action_dict['continuous_samples']
        
        return {
            'action_dict': action_dict,
            'value': value,
            'raw_outputs': raw_outputs,
            'vectorized_observation': raw_outputs.get('vectorized_observation')
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
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
        raise NotImplementedError("check that normalization of the logits is correct")
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
            
            # Apply the same clamping as in process_network_output for consistency
            continuous_log_std = torch.clamp(continuous_log_std, min=-20, max=2)
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
            discrete_distribution = torch.distributions.Categorical(logits=discrete_logits)
            discrete_action = discrete_distribution.sample()
        else:
            discrete_action = discrete_logits.argmax(dim=-1)

        
        discrete_action_reshaped = discrete_action.unsqueeze(-1)

        # Create one-hot encoding for the sampled discrete action
        discrete_one_hot = torch.zeros_like(discrete_logits).scatter_(-1, discrete_action_reshaped, 1)
        
        # Get continuous output (conditioned on sampled discrete action)
        continuous_output = self.policy.get_continuous_output(processed_obs, discrete_one_hot.squeeze(1), include_std=not self.fixed_std)
        
        if self.fixed_std:
            # Fixed std: continuous_output is just the mean
            selected_continuous_mean = continuous_output
            # Use the scalar log_std parameter
            selected_log_std = self.log_std
        else:
            # State-dependent std: continuous_output contains both mean and log_std
            n_continuous = continuous_output.size(-1) // 2
            selected_continuous_mean = continuous_output[..., :n_continuous]
            selected_log_std = continuous_output[..., n_continuous:]
        
        # First remove the extra dimension, and then expand along the features dimension
        # this is like "playing" the same continuous mean for all discrete actions
        # so that we match how we process later on (all discrete actions but one will be zero)
        selected_continuous_mean = selected_continuous_mean.squeeze(2)
        expanded_continuous_mean = selected_continuous_mean.unsqueeze(1).expand(-1, discrete_logits.size(1), discrete_logits.size(2))
        
        # Expand log_std to match the expanded mean's shape
        if self.fixed_std:
            expanded_log_std = selected_log_std.expand_as(expanded_continuous_mean)
        else:
            selected_log_std = selected_log_std.squeeze(2)
            expanded_log_std = selected_log_std.unsqueeze(1).expand(-1, discrete_logits.size(1), discrete_logits.size(2))
        
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
            argmax=False, # don't use argmax for discrete
            sample=False, # don't sample for discrete
            random_continuous=train,  # Sample for training, don't sample for inference (use mean as the continuous action)
            discrete_probs=discrete_one_hot,  # Pass the one-hot encoded discrete action
            observations=observation
        )
        
        if train:
            # Store the original selected continuous mean for PPO calculations
            action_dict['selected_continuous_mean'] = selected_continuous_mean
            action_dict['discrete_action'] = discrete_action
            
            # Calculate and store log probabilities for PPO using BaseAgent helper function
            # Use the same clamped log_std that was used for sampling to ensure consistency
            if self.fixed_std:
                clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
                continuous_std = torch.exp(clamped_log_std).expand_as(selected_continuous_mean)
            else:
                # For state-dependent std, use the selected log_std directly
                clamped_log_std = torch.clamp(selected_log_std, min=-20, max=2)
                continuous_std = torch.exp(clamped_log_std)
                
            total_log_probs = self.calculate_log_probs(
                discrete_logits,
                discrete_action,
                selected_continuous_mean,
                continuous_std,
                action_dict['raw_continuous_samples']
            )
            action_dict['log_probs'] = total_log_probs
        
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
    
    def get_log_probs_value_and_entropy(self, processed_observation, discrete_action_indices, continuous_samples=None):
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
        
        # Get continuous output conditioned on the discrete action
        continuous_output = self.policy.get_continuous_output(
            processed_observation, 
            discrete_one_hot.squeeze(1),
            include_std=not self.fixed_std
        )
        
        if self.fixed_std:
            # Fixed std: continuous_output is just the mean
            selected_continuous_mean = continuous_output.squeeze(2)
            # Use the scalar log_std parameter
            clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
            continuous_std = torch.exp(clamped_log_std).expand_as(selected_continuous_mean)
        else:
            # State-dependent std: continuous_output contains both mean and log_std
            n_continuous = continuous_output.size(-1) // 2
            selected_continuous_mean = continuous_output[..., :n_continuous].squeeze(2)
            selected_log_std = continuous_output[..., n_continuous:].squeeze(2)
            clamped_log_std = torch.clamp(selected_log_std, min=-20, max=2)
            continuous_std = torch.exp(clamped_log_std)
        
        # Use the helper function to calculate total log probabilities
        total_log_probs = self.calculate_log_probs(
            discrete_logits,
            discrete_action_indices.unsqueeze(-1),
            selected_continuous_mean,
            continuous_std,
            continuous_samples
        )
        
        # Get value if value network exists
        value = self.value_net(processed_observation, process_state=False) if self.value_net else None
        
        # Calculate total entropy
        discrete_entropy = self.get_entropy(discrete_logits)
        continuous_entropy = self.get_gaussian_entropy(clamped_log_std)
        total_entropy = discrete_entropy + continuous_entropy
        
        return total_log_probs, value, total_entropy
