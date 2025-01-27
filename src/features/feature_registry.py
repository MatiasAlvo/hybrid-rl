from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from src.envs.inventory.range_manager import RangeManager
import torch.nn.functional as F

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
        self.device = config.get('device', 'cpu')
        
        # Setup feature processing (for future PPO use)
        self.state_features = self._setup_state_features()
        self.observation_features = self._setup_observation_features()
        self.feature_normalizers = self._setup_normalizers()
    
    def _get_network_dimensions(self):
        """Get dimensions from policy configuration"""
        policy_params = self.config['policy_network']
            
        return {
            'input_size': policy_params.get('input_size', 2),
            'n_discrete': policy_params['heads']['discrete']['size'],
            'n_continuous': policy_params['heads']['continuous']['size']
        }
    
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
    
    def process_network_output(self, raw_outputs, argmax=False, sample=False):
        """Process network outputs into action space"""
        # Get discrete probabilities if discrete head exists
        discrete_probs = None
        if 'discrete' in raw_outputs:
            discrete_probs = self.range_manager.get_discrete_probabilities(
                raw_outputs['discrete'], 
                argmax=argmax,
                sample=sample
            )
            # Get indices of selected actions (where the 1s are in discrete_probs)
            discrete_actions = discrete_probs.argmax(dim=-1)  # This gives us indices instead of one-hot
            # Get logits for selected actions
            action_logits = raw_outputs['discrete'].gather(-1, discrete_actions.unsqueeze(-1)).squeeze(-1)
        
        # Get continuous values if continuous head exists
        continuous_values = None
        if 'continuous' in raw_outputs:
            # First get [0,1] values
            continuous_values = self.range_manager.get_continuous_values(
                raw_outputs['continuous']
            )
            # Then scale to actual ranges
            continuous_values = self.range_manager.scale_continuous_by_ranges(
                continuous_values,
                self.range_manager.get_continuous_ranges()
            )
        # Combine discrete and continuous actions into feature-specific actions
        feature_actions = self.range_manager.compute_feature_actions(
            discrete_probs, 
            continuous_values
        )
        
        return {
            'discrete_probs': discrete_probs,  # One-hot format
            'discrete_actions': discrete_actions,  # Index format
            'action_logits': action_logits,
            'logits': raw_outputs['discrete'],  # Full logits
            'continuous_values': continuous_values,
            'feature_actions': feature_actions
        }
    
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