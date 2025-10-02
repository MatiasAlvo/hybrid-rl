from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import numpy as np
import torch.nn.functional as F

@dataclass
class FeatureRange:
    name: str
    ranges: List[List[float]]
    values: List[float]
    
class RangeManager:
    """
    Manages ranges for hybrid actions by partitioning the action space
    based on all threshold points from all features
    """
    def __init__(self, config: Dict, device: torch.device):
        self.device = device
        self.discrete_features = self._process_discrete_features(config.get('discrete_features', {}))
        self.validate_features()
        self.breakpoints = self._compute_breakpoints()
        # print(f'breakpoint: {self.breakpoints}')
        self.ranges = self._compute_ranges()
        self.n_sub_ranges = len(self.ranges)
        
        # Pre-compute mappings from sub-ranges to feature ranges
        self.feature_range_mappings = self._precompute_feature_range_mappings()
        
        # Valid combinations are all sub-ranges when using range-based approach
        self.valid_combinations = list(range(self.n_sub_ranges)) if self.ranges else None
        
        # Determine which activation function to use for each range
        self._precompute_activation_types()
        
        # Pre-compute shift and scale factors for continuous value scaling
        self._precompute_continuous_scale_factors()
        
        # Store configuration for mean demand scaling
        self.scale_by_mean_demand = config.get('scale_by_mean_demand', False)
    
    def _process_discrete_features(self, discrete_features: Dict) -> Dict:
        """
        Process discrete features configuration to handle 'inf' strings.
        Converts any string 'inf' values to float('inf').
        """
        processed_features = {}
        
        for feature_name, feature in discrete_features.items():
            if feature is None:
                processed_features[feature_name] = None
                continue
            
            processed_feature = feature.copy()  # Create a copy to avoid modifying the original
            
            # Process thresholds to convert string 'inf' to float('inf')
            if 'thresholds' in processed_feature:
                processed_thresholds = []
                for threshold in processed_feature['thresholds']:
                    if isinstance(threshold, str) and threshold.lower() == 'inf':
                        processed_thresholds.append(float('inf'))
                    else:
                        processed_thresholds.append(threshold)
                processed_feature['thresholds'] = processed_thresholds
            
            processed_features[feature_name] = processed_feature
        
        return processed_features
    
    def validate_features(self):
        """Validate feature definitions"""
        if not self.discrete_features:
            return
            
        features = [f for f in self.discrete_features.values() if f is not None]
        if not features:
            return
            
        # All features should start at min and end at max
        global_min = min(f['thresholds'][0] for f in features)
        global_max = max(f['thresholds'][-1] for f in features)
        
        for feature in features:
            thresholds = feature['thresholds']
            values = feature['values']
            
            assert len(values) == len(thresholds) - 1, \
                f"Number of values ({len(values)}) should be one less than thresholds ({len(thresholds)})"
            assert thresholds[0] == global_min, f"All features should start at {global_min}"
            assert thresholds[-1] == global_max, f"All features should end at {global_max}"
    
    def _compute_breakpoints(self) -> np.ndarray:
        """Compute unique breakpoints from all features"""
        if not self.discrete_features:
            return np.array([])
            
        # Collect all threshold points
        all_thresholds = []
        for feature in self.discrete_features.values():
            if feature is not None:
                all_thresholds.extend(feature['thresholds'])
                
        # Get unique breakpoints and sort them
        return np.sort(np.unique(all_thresholds))
    
    def _compute_ranges(self) -> List[List[float]]:
        """Compute ranges based on breakpoints"""
        if len(self.breakpoints) < 2:
            return []
            
        return [[self.breakpoints[i], self.breakpoints[i+1]] 
                for i in range(len(self.breakpoints)-1)]
    
    def _precompute_feature_range_mappings(self) -> Dict:
        """
        Pre-compute which sub-ranges correspond to each feature range.
        Returns a dictionary of feature mappings, where each mapping contains:
        - range_indices: which sub-ranges belong to each feature range
        - values: corresponding feature values for each range
        """
        mappings = {}
        
        for feature_name, feature in self.discrete_features.items():
            if feature is None:
                continue
                
            original_ranges = list(zip(feature['thresholds'][:-1], feature['thresholds'][1:]))
            n_original_ranges = len(original_ranges)
            
            # For each original range, find which sub-ranges belong to it
            range_indices = [[] for _ in range(n_original_ranges)]
            
            for i, (orig_start, orig_end) in enumerate(original_ranges):
                for j, (sub_start, sub_end) in enumerate(self.ranges):
                    # print(f'sub_start: {sub_start}, sub_end: {sub_end}, orig_start: {orig_start}, orig_end: {orig_end}')
                    if sub_start >= orig_start and sub_end <= orig_end:
                        range_indices[i].append(j)
            
            mappings[feature_name] = {
                'range_indices': range_indices,
                'values': torch.tensor(feature['values'], device=self.device)
            }
            
        return mappings
    
    def get_network_dimensions(self) -> Dict:
        """Get dimensions needed for network outputs"""
        if not self.discrete_features:
            return {'n_discrete': 0, 'n_continuous': 1}
            
        return {
            'n_discrete': self.n_sub_ranges,    # One discrete output per sub-range
            'n_continuous': self.n_sub_ranges   # One continuous output per sub-range
        }
    
    def convert_network_output_to_simulator_action(
        self, 
        logits: torch.Tensor,    # Shape: (batch_size, n_stores, n_sub_ranges) (unnormalized real values)
        continuous_values: torch.Tensor,  # Shape: (batch_size, n_stores, n_sub_ranges) (unnormalized real values)
        use_argmax: bool = False
    ) -> Dict:
        """
        Convert network outputs to various action representations needed for
        simulation and optimization.
        """
        batch_size, n_stores, _ = logits.shape
        
        # 1. Scale continuous values to each range
        continuous_values = torch.sigmoid(continuous_values)  # Scale to [0,1]
        continuous_per_sub_range = torch.zeros_like(continuous_values)
        for i, (range_min, range_max) in enumerate(self.ranges):
            continuous_per_sub_range[..., i] = range_min + continuous_values[..., i] * (range_max - range_min)
        
        # 2. Process discrete probabilities
        discrete_probs = F.softmax(logits, dim=-1)
        if use_argmax:
            discrete_one_hot = torch.zeros_like(discrete_probs)
            max_indices = discrete_probs.argmax(dim=-1)  # Shape: (batch_size, n_stores)
            # Create indices for scatter
            batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_stores)
            store_indices = torch.arange(n_stores).view(1, -1).expand(batch_size, -1)
            discrete_one_hot[batch_indices, store_indices, max_indices] = 1.0
            discrete_probs = discrete_one_hot
        
        # 3. Map to original feature ranges using pre-computed mappings
        feature_mappings = {}
        for feature_name, mapping in self.feature_range_mappings.items():
            feature_discrete = torch.zeros(
                (batch_size, n_stores, len(mapping['values'])), 
                device=discrete_probs.device
            )
            feature_continuous = torch.zeros(
                (batch_size, n_stores, len(mapping['values'])), 
                device=continuous_per_sub_range.device
            )
            
            # For each original range of this feature
            for i, sub_range_indices in enumerate(mapping['range_indices']):
                # Sum probabilities and weighted continuous values for all sub-ranges in this range
                for j in sub_range_indices:
                    feature_discrete[..., i] += discrete_probs[..., j]
                    feature_continuous[..., i] = torch.clamp(feature_continuous[..., i] + discrete_probs[..., j] * continuous_per_sub_range[..., j], min=0)
                    # feature_continuous[..., i] += discrete_probs[..., j] * continuous_per_sub_range[..., j]
            
            feature_mappings[feature_name] = {
                'discrete': feature_discrete,      # Shape: (batch_size, n_stores, n_feature_ranges)
                'continuous': feature_continuous,  # Shape: (batch_size, n_stores, n_feature_ranges)
                'value':
                 mapping['values']
            }
        
        # 4. Calculate the total action by summing over feature_continuous for any feature
        total_action = torch.sum(feature_continuous, dim=-1)
        
        return {
            'continuous_per_sub_range': continuous_per_sub_range,  # Shape: (batch_size, n_stores, n_sub_ranges)
            'discrete_probs': discrete_probs,             # Shape: (batch_size, n_stores, n_sub_ranges)
            'feature_mappings': feature_mappings,          # Per-feature mappings with store dimension
            'total_action': total_action
        }
    
    def get_action_ranges(self):
        """Returns the ranges for each action type"""
        if not self.discrete_features:
            return {
                'discrete': None,
                'continuous': None
            }
            
        return {
            'discrete': self.ranges,  # Using self.ranges from _compute_ranges()
            'continuous': self.ranges  # Same ranges for both types
        }
    
    def get_discrete_probabilities(self, logits, argmax=False, sample=False):
        """Convert logits to probabilities or one-hot vectors"""
        if logits is None:
            return None
        
        # Debug logits before softmax
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN or Inf in logits before softmax")
            print("Logits stats:", 
                  f"range [{logits.min().item():.3f}, {logits.max().item():.3f}], "
                  f"mean {logits.mean().item():.3f}")
        
        probs = F.softmax(logits, dim=-1)
        # print(f'probs: {probs[0]}')
        
        # Debug probabilities
        if (probs < 0).any() or (probs > 1).any():
            print("Warning: Invalid probability values")
            print("Probs stats:", 
                  f"range [{probs.min().item():.3f}, {probs.max().item():.3f}], "
                  f"mean {probs.mean().item():.3f}")
        
        if argmax:
            indices = probs.argmax(dim=-1)
            probs = F.one_hot(indices, num_classes=logits.size(-1)).float()
        elif sample:
            try:
                indices = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)  # Sample from last dim
                probs = torch.zeros_like(probs)
                probs.scatter_(-1, indices.view(probs.shape[:-1] + (1,)), 1)  # Place 1s in last dim
            except RuntimeError as e:
                print("Error in multinomial sampling:")
                print("Probs shape:", probs.shape)
                print("Probs sum:", probs.sum(-1))
                print("Any NaN:", torch.isnan(probs).any())
                print("Any Inf:", torch.isinf(probs).any())
                print("Any negative:", (probs < 0).any())
                raise e
        # print(f'average probs: {probs.mean(dim=0)}')
        return probs, None
    
    def _precompute_activation_types(self):
        """
        Determine which activation function to use for each range.
        - sigmoid for ranges with finite upper bounds
        - softplus for ranges with infinite upper bounds
        """
        if not self.ranges:
            self.activation_types = None
            return
        
        activation_types = []
        for i, (_, upper_bound) in enumerate(self.ranges):
            if np.isinf(upper_bound):
                activation_types.append('softplus')
            else:
                activation_types.append('sigmoid')
        print(f'activation_types: {activation_types}')
        self.activation_types = activation_types
    
    def _precompute_continuous_scale_factors(self):
        """Pre-compute shift and scale factors for continuous value scaling"""
        if not self.ranges:
            self.post_activation_shifts = None
            self.post_activation_scales = None
            return
        
        shifts = []
        scales = []
        for i, (min_val, max_val) in enumerate(self.ranges):
            shifts.append(min_val)
            
            if self.activation_types[i] == 'sigmoid':
                # For sigmoid, scale by the range size
                scales.append(max_val - min_val)
            else:  # softplus
                # For softplus, scale by 1 since the activation already handles the scaling
                scales.append(1.0)
        
        # Use float32 explicitly to match PyTorch's default dtype
        self.post_activation_shifts = torch.tensor(shifts, device=self.device, dtype=torch.float32)
        self.post_activation_scales = torch.tensor(scales, device=self.device, dtype=torch.float32)
    
    def apply_activations(self, raw_values):
        """
        Scale continuous values using appropriate activation functions:
        - sigmoid for ranges with finite upper bounds
        - softplus for ranges with infinite upper bounds
        """
        if raw_values is None or self.activation_types is None:
            return None
        
        # Start with zeros tensor of the same shape
        activated_values = torch.zeros_like(raw_values)
        
        # Apply the appropriate activation function for each range
        for i, activation_type in enumerate(self.activation_types):
            if activation_type == 'sigmoid':
                activated_values[..., i] = torch.sigmoid(raw_values[..., i])
            else:  # softplus
                # activated_values[..., i] = raw_values[..., i]
                activated_values[..., i] = F.softplus(raw_values[..., i])
        
        return activated_values
    
    def scale_continuous_by_ranges(self, continuous_values, ranges, observations=None):
        """
        Scale activated values to actual ranges while maintaining input shape
        Args:
            continuous_values: tensor of shape [batch_size, n_stores, n_ranges]
            ranges: list of [min, max] pairs for each range
            observations: dict containing observation data (optional)
        Returns:
            scaled values with same shape as input
        """
        if continuous_values is None or ranges is None:
            return None
        
        # Apply standard scaling first
        scaled_values = continuous_values * self.post_activation_scales + self.post_activation_shifts
        
        # Apply mean demand scaling if enabled and observations provided
        if (self.scale_by_mean_demand and observations is not None and 
            'past_demands' in observations):
            # Calculate mean demand: [batch, stores, past_periods] -> [batch, stores]
            mean_demand = observations['past_demands'].mean(dim=-1)
            
            # Expand to match continuous_values shape: [batch, stores] -> [batch, stores, n_ranges]
            mean_demand_expanded = mean_demand.unsqueeze(-1).expand_as(continuous_values)
            
            # Apply mean demand scaling
            scaled_values = scaled_values * 1
            # scaled_values = scaled_values * mean_demand_expanded
        
        return scaled_values
    
    def compute_feature_actions(self, discrete_probs, continuous_values):
        """
        For each feature and each of its ranges:
        - Get the pre-computed mapping of which sub-ranges correspond to this range
        - Sum (discrete_probs * continuous_values) over those sub-ranges
        """
        if discrete_probs is None and continuous_values is None:
            return None
            
        feature_actions = {}
        batch_size, n_stores, _ = discrete_probs.shape
        
        # Iterate over features
        for feature_name, mapping in self.feature_range_mappings.items():
            n_feature_ranges = len(mapping['values'])
            feature_action = torch.zeros(batch_size, n_stores, n_feature_ranges, device=discrete_probs.device)
            feature_discrete = torch.zeros(batch_size, n_stores, n_feature_ranges, device=discrete_probs.device)
            
            # For each range of this feature
            for range_idx, sub_range_indices in enumerate(mapping['range_indices']):

                # In range_manager.py around line 376, replace the failing line with:

                if (torch.isnan(discrete_probs).any() or torch.isnan(continuous_values).any() or 
                    torch.isinf(discrete_probs).any() or torch.isinf(continuous_values).any()):
                    
                    print("=== NaN/Inf DETECTED IN RANGE_MANAGER ===")
                    print(f"discrete_probs has NaN: {torch.isnan(discrete_probs).any()}")
                    print(f"discrete_probs has Inf: {torch.isinf(discrete_probs).any()}")
                    print(f"continuous_values has NaN: {torch.isnan(continuous_values).any()}")
                    print(f"continuous_values has Inf: {torch.isinf(continuous_values).any()}")
                    print(f"discrete_probs shape: {discrete_probs.shape}")
                    print(f"continuous_values shape: {continuous_values.shape}")
                    print(f"sub_range_indices: {sub_range_indices}")
                    # raise ValueError("NaN/Inf detected before range_manager operation")

                # Check for index bounds (fixed for list)
                if isinstance(sub_range_indices, list) and sub_range_indices:
                    max_idx = max(sub_range_indices)
                    min_idx = min(sub_range_indices)
                    if (max_idx >= discrete_probs.shape[-1] or 
                        max_idx >= continuous_values.shape[-1] or 
                        min_idx < 0):
                        
                        print("=== INDEX OUT OF BOUNDS DETECTED ===")
                        print(f"sub_range_indices range: [{min_idx}, {max_idx}]")
                        print(f"discrete_probs max valid index: {discrete_probs.shape[-1] - 1}")
                        print(f"continuous_values max valid index: {continuous_values.shape[-1] - 1}")
                        # raise ValueError("Index out of bounds in range_manager")

                # Original operation
                sub_actions = discrete_probs[..., sub_range_indices] * continuous_values[..., sub_range_indices]

                # Sum over the sub-ranges that correspond to this range
                # (this correspondence was pre-computed in _precompute_feature_range_mappings)
                sub_actions = discrete_probs[..., sub_range_indices] * continuous_values[..., sub_range_indices]
                feature_action[..., range_idx] = torch.sum(sub_actions, dim=-1, keepdim=False)
                # feature_action[..., range_idx] = torch.clamp(torch.sum(sub_actions, dim=-1, keepdim=False), min=0)

                # feature_action[..., range_idx] = torch.sum(sub_actions, dim=-1, keepdim=False)
                feature_discrete[..., range_idx] = torch.sum(discrete_probs[..., sub_range_indices], dim=-1, keepdim=False)
            feature_actions[feature_name] = {
                'action': feature_action,  # Shape: [batch_size, n_stores, n_feature_ranges] (represents the action for each range)
                'range_probs': feature_discrete,  # Shape: [batch_size, n_stores, n_feature_ranges] (represents the discrete probabilities for each range)
                'values': mapping['values']
            }
        
        feature_actions['total_action'] = self.compute_total_action(discrete_probs, continuous_values, non_negative=False)
        
        return feature_actions

    def compute_total_action(self, discrete_probs, continuous_values, non_negative=True):
        """
        Compute the total action as the sum of discrete probabilities multiplied by continuous values.
        
        Args:
            discrete_probs: tensor of shape [batch_size, n_stores, n_discrete]
            continuous_values: tensor of shape [batch_size, n_stores, n_continuous]
        
        Returns:
            total_action: tensor of shape [batch_size, n_stores, n_total_actions]
        """
        if discrete_probs is None or continuous_values is None:
            return None
        
        # Element-wise multiplication and summation across the last dimension
        total_action = discrete_probs * continuous_values
        
        # Sum across the continuous dimension to get total action
        total_action = total_action.sum(dim=-1)  # Shape: [batch_size, n_stores]
        
        if non_negative:
            total_action = torch.clamp(total_action, min=0)
        
        return total_action
    
    def get_continuous_ranges(self):
        """Returns the ranges for continuous actions"""
        if not self.discrete_features:
            return None
        return self.ranges  # Using self.ranges from _compute_ranges()
    
    def get_discrete_ranges(self):
        """Returns the ranges for discrete actions"""
        if not self.discrete_features:
            return None
        return self.ranges  # Same ranges for both types