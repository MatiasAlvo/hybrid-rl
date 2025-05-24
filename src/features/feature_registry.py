from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
from src.envs.inventory.range_manager import RangeManager
import torch.nn.functional as F
import torch.distributions as dist

@dataclass
class FeatureRange:
    name: str
    ranges: List[List[float]]
    values: List[float]

class FeatureRegistry:
    """Registry for managing feature dimensions and transformations"""
    def __init__(self, config, range_manager):
        self.config = config
        self.range_manager = range_manager
        self.network_dims = self._get_network_dimensions()
        
        # Setup feature processing (for future PPO use)
        self.state_features = self._setup_state_features()
        self.observation_features = self._setup_observation_features()
        self.feature_normalizers = self._setup_normalizers()
    
    def _get_network_dimensions(self):
        """Get dimensions from policy configuration"""
        policy_params = self.config['policy_network']
        
        # Handle different possible head configurations
        heads = policy_params.get('heads', {})
        
        dimensions = {
            'input_size': policy_params.get('input_size', 2),
            'n_discrete': 0,
            'n_continuous': 0
        }
        
        # Check for discrete head
        if 'discrete' in heads:
            dimensions['n_discrete'] = heads['discrete'].get('size', 0)
            
        # Check for continuous head
        if 'continuous' in heads:
            dimensions['n_continuous'] = heads['continuous'].get('size', 0)
            
        # Check for continuous_mean and continuous_log_std (GaussianPPOAgent)
        if 'continuous_mean' in heads:
            dimensions['n_continuous'] = heads['continuous_mean'].get('size', 0)
            
        return dimensions
    
    def get_network_dimensions(self):
        """Return network dimensions"""
        return self.network_dims
    
    def get_simulator_config(self):
        """Return simulator configuration"""
        return {
            'type': 'hybrid', 
            'discrete_size': self.network_dims['n_discrete'],
            'continuous_size': self.network_dims['n_continuous'],
            'input_size': self.network_dims['input_size']
        }
    
    def get_observation_keys(self):
        """Return observation keys for value network"""
        # You might want to customize this based on your specific environment
        return ['store_inventories']

    def process_discrete_output(self, raw_discrete_logits, argmax=False, sample=False, straight_through=False):
        """
        Process network outputs into action space for discrete actions only.
        Uses PyTorch's Categorical distribution for better compatibility with RL algorithms.
        
        Args:
            raw_discrete_logits: Tensor containing unnormalized logits
            argmax: Whether to take argmax for discrete actions
            sample: Whether to sample from discrete distribution
            straight_through: Whether to apply straight-through gradient estimation
        
        Returns:
            Dictionary containing processed discrete actions
        """
        # Get original shape for reshaping later
        original_shape = raw_discrete_logits.shape
        
        # Reshape logits to 2D for Categorical (batch_size*n_stores, n_discrete)
        reshaped_logits = raw_discrete_logits.reshape(-1, raw_discrete_logits.size(-1))
        
        # Create categorical distribution
        distribution = torch.distributions.Categorical(logits=reshaped_logits)
        
        # Get probabilities
        probs = distribution.probs
        
        # Select actions based on argmax or sampling
        if argmax:
            action_indices = probs.argmax(dim=-1)
        elif sample:
            action_indices = distribution.sample()
        else:
            # Default to using probabilities directly
            discrete_probs = probs.reshape(original_shape)
            discrete_action_indices = discrete_probs.argmax(dim=-1)
            log_probs = None
            
            # Apply straight-through gradient estimation if requested
            if straight_through:
                # Get hard one-hot encoding (forward pass)
                indices = torch.argmax(discrete_probs, dim=-1, keepdim=True)
                hard_probs = torch.zeros_like(discrete_probs).scatter_(-1, indices, 1.0)
                
                # Straight-through trick: hard values in forward pass, but gradients flow through soft values
                discrete_probs = hard_probs - discrete_probs.detach() + discrete_probs
            
            return {
                'discrete_probs': discrete_probs,  # Probability distribution
                'discrete_action_indices': discrete_action_indices,  # Index format
                'log_probs': log_probs  # Log probabilities of selected actions
            }
        
        # Get log probabilities of selected actions
        log_probs = distribution.log_prob(action_indices)
        
        # Create one-hot vectors for selected actions
        discrete_probs = torch.zeros_like(reshaped_logits)
        discrete_probs.scatter_(-1, action_indices.unsqueeze(-1), 1.0)
        
        # Reshape back to original shape
        discrete_probs = discrete_probs.reshape(original_shape)
        
        # Reshape action indices and log probs to match expected output shape
        discrete_action_indices = action_indices.reshape(original_shape[:-1])
        log_probs = log_probs.reshape(original_shape[:-1])
        
        # Apply straight-through gradient estimation if requested
        if straight_through and not argmax:  # Only apply when not using argmax
            # Get hard one-hot encoding (forward pass)
            indices = torch.argmax(discrete_probs, dim=-1, keepdim=True)
            hard_probs = torch.zeros_like(discrete_probs).scatter_(-1, indices, 1.0)
            
            # Straight-through trick: hard values in forward pass, but gradients flow through soft values
            discrete_probs = hard_probs - discrete_probs.detach() + discrete_probs
        
        # Create action dictionary
        action_dict = {
            'discrete_probs': discrete_probs,  # One-hot format
            'discrete_action_indices': discrete_action_indices,  # Index format
            'log_probs': log_probs  # Log probabilities of selected actions
        }
        
        return action_dict
    
    def process_network_output(self, raw_outputs, argmax=False, sample=False, random_continuous=False, straight_through=False, discrete_probs=None, log_probs=None):
        """
        Process network outputs into action space - works for all agent types
        
        Args:
            raw_outputs: Dictionary of raw network outputs
            argmax: Whether to take argmax for discrete actions
            sample: Whether to sample from discrete distribution
            random_continuous: Whether to sample from Gaussian distribution for continuous actions
            straight_through: Whether to apply straight-through gradient estimation
            discrete_probs: Optional pre-computed discrete probabilities
            log_probs: Optional pre-computed log probabilities
        """
        # Get discrete probabilities if discrete head exists
        discrete_action_indices = None
        logits = None
        
        if 'discrete' in raw_outputs:
            # For FactoredGaussianPPOAgent, one-hot is already passed, so we should not compute it
            if discrete_probs is None:
                # Apply softmax, then argmax if argmax is True, or sample if sample is True (and return one-hot)
                discrete_probs, log_probs = self.range_manager.get_discrete_probabilities(
                    raw_outputs['discrete'], 
                    argmax=argmax,
                    sample=sample
                )
            
                # Apply straight-through gradient estimation if requested
                if straight_through and not argmax:  # Only apply when not using argmax
                    # Get hard one-hot encoding (forward pass)
                    indices = torch.argmax(discrete_probs, dim=-1, keepdim=True)
                    hard_probs = torch.zeros_like(discrete_probs).scatter_(-1, indices, 1.0)
                    
                    # Straight-through trick: hard values in forward pass, but gradients flow through soft values
                    discrete_probs = hard_probs - discrete_probs.detach() + discrete_probs
                
            # Get indices of selected actions (where the 1s are in discrete_probs)
            discrete_action_indices = discrete_probs.argmax(dim=-1)  # This gives us indices instead of one-hot
            # Get logits for selected actions - using gather with proper reshaping
            logits = raw_outputs['discrete'].gather(-1, discrete_action_indices.unsqueeze(-1)).squeeze(-1)
        
        # Get continuous values if continuous head exists
        continuous_values = None
        raw_continuous_samples = None
        
        # Handle continuous values (standard or Gaussian)
        if 'continuous' in raw_outputs:
            raw_continuous_samples = raw_outputs['continuous']
            
            # If random_continuous is True and we have mean/std, override with sampled values
            if random_continuous and 'continuous_mean' in raw_outputs and 'continuous_log_std' in raw_outputs:
                continuous_mean = raw_outputs['continuous_mean']
                continuous_log_std = torch.clamp(raw_outputs['continuous_log_std'], min=-20, max=2)
                continuous_std = torch.exp(continuous_log_std)
                
                # Sample from the Gaussian distribution - ensure all tensors are on the same device
                epsilon = torch.randn_like(continuous_mean, device=continuous_mean.device)
                raw_continuous_samples = continuous_mean + continuous_std * epsilon
            
            # Process the continuous values
            continuous_values = self.range_manager.get_scaled_continuous_values(raw_continuous_samples)
            continuous_values = self.range_manager.scale_continuous_by_ranges(
                continuous_values,
                self.range_manager.get_continuous_ranges()
            )
        
        # Combine discrete and continuous actions into feature-specific actions
        feature_actions = self.range_manager.compute_feature_actions(
            discrete_probs, 
            continuous_values
        )
        
        # If using random_continuous (GaussianPPO), add continuous log probs of selected actions
        if random_continuous and 'continuous_mean' in raw_outputs and 'continuous_log_std' in raw_outputs:
            continuous_mean = raw_outputs['continuous_mean']
            continuous_log_std = torch.clamp(raw_outputs['continuous_log_std'], min=-20, max=2)
            continuous_std = torch.exp(continuous_log_std)
            
            # Get log probs for all continuous actions
            normal_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            continuous_log_probs = normal_dist.log_prob(raw_continuous_samples)  # [batch, n_discrete, n_continuous]
            
            # Add selected continuous log probs to discrete logits
            logits = logits + continuous_log_probs.gather(-1, discrete_action_indices.unsqueeze(-1)).squeeze(-1)
        
        # Create action dictionary with common format for all agent types
        action_dict = {
            'discrete_probs': discrete_probs,  # One-hot format
            'discrete_action_indices': discrete_action_indices,  # Index format
            'logits': logits,  # Combined logits of selected actions (both discrete and continuous)
            'continuous_values': continuous_values,
            'feature_actions': feature_actions
        }
        
        # Include raw continuous samples when using random_continuous
        if random_continuous and raw_continuous_samples is not None:
            action_dict['raw_continuous_samples'] = raw_continuous_samples
        
        return action_dict
    
    def process_continuous_output(self, raw_continuous, discrete_action_indices=None, continuous_mean=None, 
                                 continuous_log_std=None, random_continuous=False):
        """
        Process network outputs for continuous actions.
        
        Args:
            raw_continuous: Tensor containing raw continuous values
            discrete_action_indices: Indices of selected discrete actions (for combining log probs)
            continuous_mean: Mean values for Gaussian distribution (if using stochastic policy)
            continuous_log_std: Log standard deviation for Gaussian distribution (if using stochastic policy)
            random_continuous: Whether to sample from Gaussian distribution
        
        Returns:
            Dictionary containing processed continuous actions and related information
        """
        # Initialize raw_continuous_samples with the provided raw_continuous
        raw_continuous_samples = raw_continuous
        continuous_log_probs = None
        
        # If using Gaussian policy, sample from the distribution
        if random_continuous and continuous_mean is not None and continuous_log_std is not None:
            # Clamp log_std for numerical stability
            continuous_log_std = torch.clamp(continuous_log_std, min=-20, max=2)
            continuous_std = torch.exp(continuous_log_std)
            
            # Sample from the Gaussian distribution
            epsilon = torch.randn_like(continuous_mean, device=continuous_mean.device)
            raw_continuous_samples = continuous_mean + continuous_std * epsilon
            
            # Calculate log probabilities for the sampled actions
            normal_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            continuous_log_probs = normal_dist.log_prob(raw_continuous_samples)  # [batch, n_discrete, n_continuous]
            
            # If discrete_action_indices is provided, get log probs for selected actions
            if discrete_action_indices is not None:
                # Reshape indices for gathering
                gather_indices = discrete_action_indices.unsqueeze(-1).expand(-1, -1, continuous_log_probs.size(-1))
                
                # Get log probs for selected discrete actions
                selected_continuous_log_probs = continuous_log_probs.gather(1, gather_indices)
                
                # Sum log probs across continuous dimensions
                continuous_log_probs = selected_continuous_log_probs.sum(dim=-1)
        
        # Process the continuous values through range scaling
        continuous_values = self.range_manager.get_scaled_continuous_values(raw_continuous_samples)
        continuous_values = self.range_manager.scale_continuous_by_ranges(
            continuous_values,
            self.range_manager.get_continuous_ranges()
        )
        
        # Create result dictionary
        result = {
            'continuous_values': continuous_values,
            'raw_continuous_samples': raw_continuous_samples,
            'continuous_log_probs': continuous_log_probs
        }
        
        return result
    
    def process_continuous_only_output(self, continuous_values, temperature=0.5, argmax=False, straight_through=False, zero_out_indices=None, train=True):
        """
        Process the output of a ContinuousOnlyAgent that only produces continuous values.
        Creates an implied discrete distribution using the sigmoid difference approach.
        
        Args:
            continuous_values: Tensor containing the continuous action values
            temperature: Temperature parameter controlling the steepness of sigmoid transitions
            argmax: When True, convert discrete probabilities to one-hot vectors
            straight_through: Whether to apply straight-through gradient estimation
            zero_out_indices: List of indices to zero-out in repeated_continuous
            train: Whether we are in training mode
        """
        # Scale continuous values to [0, 1] range
        scaled_continuous = torch.sigmoid(continuous_values)
        
        # Get the overall range from the first and last breakpoints
        breakpoints = self.range_manager.breakpoints
        if len(breakpoints) < 2:
            return None
        
        min_val = breakpoints[0]
        max_val = breakpoints[-1]
        
        # Scale to the overall range
        scaled_continuous_values = min_val + scaled_continuous * (max_val - min_val)
        
        # Create implied discrete probabilities using sigmoid difference approach
        discrete_probs = self._create_implied_discrete_probabilities(
            scaled_continuous_values, temperature
        )
        
        # Apply straight-through gradient estimation if requested
        if straight_through and not argmax:  # Only apply when not using argmax
            # Get hard one-hot encoding (forward pass)
            indices = torch.argmax(discrete_probs, dim=-1, keepdim=True)
            hard_probs = torch.zeros_like(discrete_probs).scatter_(-1, indices, 1.0)
            
            # Straight-through trick: hard values in forward pass, but gradients flow through soft values
            discrete_probs = hard_probs - discrete_probs.detach() + discrete_probs
        # Apply argmax if requested (for inference/evaluation)
        elif argmax:
            # Get indices of maximum probabilities
            indices = discrete_probs.argmax(dim=-1, keepdim=True)
            # Create one-hot vectors
            discrete_probs = torch.zeros_like(discrete_probs).scatter_(-1, indices, 1.0)
        
        # Duplicate the continuous values for each sub-range
        n_sub_ranges = len(breakpoints) - 1
        
        # Use expand by default for memory efficiency
        repeated_continuous = scaled_continuous_values.expand(scaled_continuous_values.shape[0], scaled_continuous_values.shape[1], n_sub_ranges)
        
        # If we need to zero out indices during inference, switch to repeat to create a new tensor
        if zero_out_indices is not None and not train:
            # Print warning only once using a class attribute
            if not hasattr(self, '_zero_out_warning_printed'):
                print(f"Warning: Zeroing out indices {zero_out_indices} during inference")
                print(f"Breakpoints: {breakpoints}")
                self._zero_out_warning_printed = True
            
            # Create a new tensor using repeat instead of expand
            repeated_continuous = scaled_continuous_values.repeat(1, 1, n_sub_ranges)
            for idx in zero_out_indices:
                if idx < repeated_continuous.shape[-1]:
                    repeated_continuous[:, :, idx] = 0.0
        
        # Now that we have both discrete_probs and repeated_continuous, compute feature actions
        feature_actions = self.range_manager.compute_feature_actions(
            discrete_probs, repeated_continuous
        )
        
        # Create action dictionary with same format as process_network_output
        action_dict = {
            'discrete_probs': discrete_probs,
            'discrete_action_indices': discrete_probs.argmax(dim=-1) if discrete_probs is not None else None,
            'logits': None,  # No discrete logits in this approach
            'continuous_values': repeated_continuous,
            'feature_actions': feature_actions
        }
        
        return action_dict
    
    def _initialize_sigmoid_scaling(self, device=None):
        """
        Initialize scaling factors and masks for efficient sigmoid probability calculation.
        This should be called once during agent initialization.
        """
        # Get breakpoints from range_manager
        breakpoints = self.range_manager.breakpoints
        
        if len(breakpoints) < 2:
            self.sigmoid_scaling_initialized = False
            return
        
        # Handle duplicate breakpoints
        epsilon = 1e-5
        shifted_breakpoints = []
        prev_point = None
        
        for point in breakpoints:
            if prev_point is not None and point == prev_point:
                shifted_breakpoints.append(point - epsilon)
                shifted_breakpoints.append(point + epsilon)
            else:
                shifted_breakpoints.append(point)
            prev_point = point
        
        # Convert to tensor and store
        self.breakpoints = torch.tensor(shifted_breakpoints, device=device)
        
        # Compute range lengths and scaling factors
        n_ranges = len(self.breakpoints) - 1
        range_lengths = self.breakpoints[1:] - self.breakpoints[:-1]
        max_range_length = torch.max(range_lengths)
        
        # Create scaling factors for each breakpoint
        breakpoint_scalings = torch.ones(len(self.breakpoints), device=device)
        
        # For internal breakpoints, use scaling from the next range
        for i in range(1, len(self.breakpoints) - 1):
            next_range_length = range_lengths[i]
            breakpoint_scalings[i] = max_range_length / next_range_length
        
        # For first and last breakpoints, use scaling from adjacent range
        breakpoint_scalings[0] = max_range_length / range_lengths[0]
        breakpoint_scalings[-1] = max_range_length / range_lengths[-1]
        
        # Store these values for later use
        self.n_ranges = n_ranges
        self.breakpoint_scalings = breakpoint_scalings
        
        # Create left/right index tensors for vectorized operations
        # [0, 1, 2, ..., n_ranges-1] for left breakpoints
        self.left_idxs = torch.arange(n_ranges, device=device)
        # [1, 2, 3, ..., n_ranges] for right breakpoints
        self.right_idxs = torch.arange(1, n_ranges+1, device=device)
        
        # Create masks for special cases
        self.is_leftmost_range = torch.zeros(n_ranges, dtype=torch.bool, device=device)
        self.is_leftmost_range[0] = True
        
        self.is_rightmost_range = torch.zeros(n_ranges, dtype=torch.bool, device=device)
        self.is_rightmost_range[-1] = True
        
        # Mark initialization as complete
        self.sigmoid_scaling_initialized = True
        
    def _create_implied_discrete_probabilities(self, continuous_values, temperature):
        """
        Create implied discrete probabilities from continuous values using scaled sigmoid differences.
        Uses the same scaling logic as the visualization routine for consistency.
        
        Args:
            continuous_values: Tensor containing the continuous action values
            temperature: Temperature parameter controlling the steepness of sigmoid transitions
            
        Returns:
            discrete_probs: Tensor containing the implied discrete probabilities
        """
        # Check if initialization has been done
        if not hasattr(self, 'sigmoid_scaling_initialized') or not self.sigmoid_scaling_initialized:
            self._initialize_sigmoid_scaling(device=continuous_values.device)
            if not self.sigmoid_scaling_initialized:
                return None
        
        # Ensure continuous_values is float32 to match model weights
        continuous_values = continuous_values.float()
        
        # Get breakpoints from range_manager
        breakpoints = self.range_manager.breakpoints
        
        if len(breakpoints) < 2:
            return None
        
        # Handle duplicate breakpoints by shifting with a small epsilon
        epsilon = 1e-5
        shifted_breakpoints = []
        prev_point = None
        
        for point in breakpoints:
            if prev_point is not None and point == prev_point:
                # If this breakpoint is the same as the previous one, shift it slightly
                shifted_breakpoints.append(point - epsilon)
                shifted_breakpoints.append(point + epsilon)
            else:
                shifted_breakpoints.append(point)
            prev_point = point
        
        # Convert to tensor with explicit float type
        breakpoints = torch.tensor(shifted_breakpoints, device=continuous_values.device, dtype=torch.float32)
        
        # Get original shape of continuous values
        batch_shape = continuous_values.shape[:-1]
        n_ranges = len(breakpoints) - 1
        
        # Compute range lengths - same as in visualization
        range_lengths = breakpoints[1:] - breakpoints[:-1]
        max_range_length = torch.max(range_lengths)
        
        # Create scaling factors for each breakpoint - same as in visualization
        breakpoint_scalings = torch.ones(len(breakpoints), device=continuous_values.device)
        
        # For internal breakpoints, use scaling from the next range
        for i in range(1, len(breakpoints) - 1):
            next_range_length = range_lengths[i]
            breakpoint_scalings[i] = max_range_length / next_range_length
        
        # For first and last breakpoints, use scaling from adjacent range
        breakpoint_scalings[0] = max_range_length / range_lengths[0]
        breakpoint_scalings[-1] = max_range_length / range_lengths[-1]
        
        # Create a tensor to store the weights for each range
        weights = torch.zeros((*batch_shape, n_ranges), device=continuous_values.device, dtype=torch.float32)
        
        # Reshape continuous_values for broadcasting
        # Add singleton dimension at the end to make it compatible with range-wise operations
        continuous_values_expanded = continuous_values.unsqueeze(-1)
        
        # Calculate sigmoid difference for each range with proper scaling
        for i in range(n_ranges):
            a_k = breakpoints[i]
            a_k_plus_1 = breakpoints[i+1]
            
            # Get scaling factors for left and right breakpoints of this range
            left_scaling = breakpoint_scalings[i]
            right_scaling = breakpoint_scalings[i+1]
            
            # Calculate sigmoid values - same logic as visualization
            if i == 0:
                # For leftmost range, left sigmoid is always 1
                left_sigmoid = torch.ones_like(continuous_values)
                right_sigmoid = torch.sigmoid((right_scaling * (continuous_values - a_k_plus_1)) / temperature)
            elif i == n_ranges - 1:
                # For rightmost range, right sigmoid is always 0
                left_sigmoid = torch.sigmoid((left_scaling * (continuous_values - a_k)) / temperature)
                right_sigmoid = torch.zeros_like(continuous_values)
            else:
                # Normal case for middle ranges
                left_sigmoid = torch.sigmoid((left_scaling * (continuous_values - a_k)) / temperature)
                right_sigmoid = torch.sigmoid((right_scaling * (continuous_values - a_k_plus_1)) / temperature)
            
            # Calculate sigmoid difference
            sigmoid_diff = left_sigmoid - right_sigmoid
            
            # Store the clamped weights
            weights[:, :, i] = torch.clamp(sigmoid_diff[:, :, 0], 0.0, 1.0)
        
        # Normalize to ensure we have a valid probability distribution
        total_weights = weights.sum(dim=-1, keepdim=True)
        discrete_probs = weights / (total_weights + 1e-10)
        
        return discrete_probs

    def _create_implied_discrete_probabilities_v1(self, continuous_values, temperature):
        """
        Create implied discrete probabilities from continuous values using scaled sigmoid differences.
        Uses pre-computed scaling factors and vectorized operations for efficiency.
        
        Args:
            continuous_values: Tensor containing the continuous action values
            temperature: Temperature parameter controlling the steepness of sigmoid transitions
            
        Returns:
            discrete_probs: Tensor containing the implied discrete probabilities
        """
        # Check if initialization has been done
        if not hasattr(self, 'sigmoid_scaling_initialized') or not self.sigmoid_scaling_initialized:
            self._initialize_sigmoid_scaling(device=continuous_values.device)
            if not self.sigmoid_scaling_initialized:
                return None
        
        # Ensure continuous_values is float32 to match model weights
        continuous_values = continuous_values.float()
        
        # Get breakpoints from range_manager
        breakpoints = self.range_manager.breakpoints
        
        if len(breakpoints) < 2:
            return None
        
        # Handle duplicate breakpoints by shifting with a small epsilon
        epsilon = 1e-5
        shifted_breakpoints = []
        prev_point = None
        
        for point in breakpoints:
            if prev_point is not None and point == prev_point:
                # If this breakpoint is the same as the previous one, shift it slightly
                shifted_breakpoints.append(point - epsilon)
                shifted_breakpoints.append(point + epsilon)
            else:
                shifted_breakpoints.append(point)
            prev_point = point
        
        # Convert to tensor with explicit float type
        shifted_breakpoints = torch.tensor(shifted_breakpoints, device=continuous_values.device, dtype=torch.float32)
        
        # Get original shape of continuous values
        batch_shape = continuous_values.shape[:-1]
        n_sub_ranges = len(shifted_breakpoints) - 1
        
        # Create a tensor to store the sigmoid differences for each sub-range
        discrete_probs = torch.zeros((*batch_shape, n_sub_ranges), device=continuous_values.device, dtype=torch.float32)
        
        # Calculate sigmoid difference for each sub-range
        for i in range(n_sub_ranges):
            a_k = shifted_breakpoints[i]
            a_k_plus_1 = shifted_breakpoints[i+1]
            
            # Special case for left-most sub-range
            if i == 0:
                # Use right sigmoid only (assume left sigmoid is always 1)
                right_sigmoid = torch.sigmoid((continuous_values - a_k_plus_1) / temperature)
                sigmoid_diff = 1.0 - right_sigmoid
            # Special case for right-most sub-range
            elif i == n_sub_ranges - 1:
                # Use left sigmoid only (assume right sigmoid is always 0)
                left_sigmoid = torch.sigmoid((continuous_values - a_k) / temperature)
                sigmoid_diff = left_sigmoid
            # Normal case for middle sub-ranges
            else:
                # Calculate sigmoid difference for this range
                left_sigmoid = torch.sigmoid((continuous_values - a_k) / temperature)
                right_sigmoid = torch.sigmoid((continuous_values - a_k_plus_1) / temperature)
                sigmoid_diff = left_sigmoid - right_sigmoid
            
            # Calculate the scaling factor to ensure maximum weight is 1
            d = a_k_plus_1 - a_k
            scaling_factor = 1.0 / (1.0 - 2.0 * torch.sigmoid(-d / (2.0 * temperature)))
            
            # Apply scaling factor to ensure maximum is 1
            scaled_sigmoid_diff = sigmoid_diff * scaling_factor
            
            # Store in the discrete_probs tensor
            discrete_probs[..., i] = scaled_sigmoid_diff[..., 0]
        
        # Clamp values to be within [0, 1] to handle potential numerical issues
        discrete_probs = torch.clamp(discrete_probs, 0.0, 1.0)
        
        # Normalize to ensure we have a valid probability distribution
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        return discrete_probs

    def _create_implied_discrete_probabilities_old(self, continuous_values, temperature):
        """
        Create implied discrete probabilities from continuous values using sigmoid difference.
        
        Args:
            continuous_values: Tensor containing the scaled continuous action values
            temperature: Temperature parameter controlling the steepness of sigmoid transitions
            
        Returns:
            discrete_probs: Tensor containing the implied discrete probabilities
        """
        # Get breakpoints from range_manager
        breakpoints = self.range_manager.breakpoints
        
        if len(breakpoints) < 2:
            return None
        
        # Handle duplicate breakpoints by shifting with a small epsilon
        epsilon = 1e-5
        shifted_breakpoints = []
        prev_point = None
        
        for point in breakpoints:
            if prev_point is not None and point == prev_point:
                # If this breakpoint is the same as the previous one, shift it slightly
                shifted_breakpoints.append(point - epsilon)
                shifted_breakpoints.append(point + epsilon)
            else:
                shifted_breakpoints.append(point)
            prev_point = point
        
        # Convert to tensor
        shifted_breakpoints = torch.tensor(shifted_breakpoints, device=continuous_values.device)
        
        # Get original shape of continuous values
        batch_shape = continuous_values.shape[:-1]
        n_sub_ranges = len(shifted_breakpoints) - 1
        
        # Create a tensor to store the sigmoid differences for each sub-range
        discrete_probs = torch.zeros((*batch_shape, n_sub_ranges), device=continuous_values.device)
        
        # Calculate sigmoid difference for each sub-range
        for i in range(n_sub_ranges):
            a_k = shifted_breakpoints[i]
            a_k_plus_1 = shifted_breakpoints[i+1]
            
            # Special case for left-most sub-range
            if i == 0:
                # Use right sigmoid only (assume left sigmoid is always 1)
                right_sigmoid = torch.sigmoid((continuous_values - a_k_plus_1) / temperature)
                sigmoid_diff = 1.0 - right_sigmoid
            # Special case for right-most sub-range
            elif i == n_sub_ranges - 1:
                # Use left sigmoid only (assume right sigmoid is always 0)
                left_sigmoid = torch.sigmoid((continuous_values - a_k) / temperature)
                sigmoid_diff = left_sigmoid
            # Normal case for middle sub-ranges
            else:
                # Calculate sigmoid difference for this range
                left_sigmoid = torch.sigmoid((continuous_values - a_k) / temperature)
                right_sigmoid = torch.sigmoid((continuous_values - a_k_plus_1) / temperature)
                sigmoid_diff = left_sigmoid - right_sigmoid
            
            # Calculate the scaling factor to ensure maximum weight is 1
            d = a_k_plus_1 - a_k
            scaling_factor = 1.0 / (1.0 - 2.0 * torch.sigmoid(-d / (2.0 * temperature)))
            
            # Apply scaling factor to ensure maximum is 1
            scaled_sigmoid_diff = sigmoid_diff * scaling_factor
            
            # Store in the discrete_probs tensor
            discrete_probs[..., i] = scaled_sigmoid_diff[..., 0]
        
        # Clamp values to be within [0, 1] to handle potential numerical issues
        discrete_probs = torch.clamp(discrete_probs, 0.0, 1.0)
        
        # Normalize to ensure we have a valid probability distribution
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        return discrete_probs
    
    def process_state(self, state):
        return state
        """Process state features for network input"""
        # Will be implemented for PPO
        processed_state = self._apply_state_features(state)
        normalized_state = self._normalize_features(processed_state)
        return normalized_state.to(self.device)
    
    # Helper methods (to be implemented for PPO)
    def _to_one_hot(self, probs):
        """Convert probabilities to one-hot vectors"""
        indices = probs.argmax(dim=-1)
        return F.one_hot(indices, num_classes=probs.size(-1)).float()
    
    def _setup_state_features(self):
        """Setup state feature processors"""
        # Will be implemented for PPO
        return {}
    
    def _setup_observation_features(self):
        """Setup observation feature processors"""
        # Will be implemented for PPO
        return {}
    
    def _setup_normalizers(self):
        """Setup feature normalizers"""
        # Will be implemented for PPO
        return {}
    
    def _apply_state_features(self, state):
        """Apply state feature processing"""
        # Will be implemented for PPO
        return state
    
    def _normalize_features(self, features):
        """Apply feature normalization"""
        # Will be implemented for PPO
        return features

    def compute_feature_actions_from_outputs(self, discrete_probs, continuous_values):
        """
        Compute feature actions from discrete probabilities and continuous values.
        This is a wrapper around the range_manager's compute_feature_actions method.
        
        Args:
            discrete_probs: Tensor containing discrete probabilities
            continuous_values: Tensor containing continuous values
            
        Returns:
            Dictionary containing feature actions
        """
        # Compute feature actions using range manager
        feature_actions = self.range_manager.compute_feature_actions(
            discrete_probs, 
            continuous_values
        )
        
        return feature_actions
