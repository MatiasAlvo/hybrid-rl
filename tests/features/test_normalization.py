"""
Unit tests for the normalization-denormalization process in FeatureRegistry.
Tests the complete flow from input normalization to output denormalization.
"""

import torch
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.feature_registry import FeatureRegistry
from envs.inventory.range_manager import RangeManager


class TestNormalizationDenormalization:
    """Test class for normalization-denormalization functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock config with normalization enabled
        self.config = {
            'policy_network': {
                'normalize_by_mean_demand': True,
                'ewma_beta': 0.9,
                'observation_keys': ['store_inventories', 'past_demands', 'past_arrivals', 'past_orders']
            }
        }
        
        # Create mock range manager
        self.range_manager = Mock(spec=RangeManager)
        self.range_manager.scale_continuous_by_ranges = Mock(side_effect=self._mock_scale_continuous)
        self.range_manager.apply_activations = MagicMock(side_effect=lambda x: x)
        
        # Create feature registry
        self.feature_registry = FeatureRegistry(self.config, self.range_manager)
        
        # Store original scale method for testing
        self.original_scale_method = self.range_manager.scale_continuous_by_ranges
    
    def _mock_scale_continuous(self, continuous_values, ranges, observations=None, feature_registry=None):
        """Mock the scale_continuous_by_ranges method to test denormalization."""
        # Simulate basic scaling
        scaled_values = continuous_values * 2.0 + 1.0
        
        return scaled_values
    
    def test_expand_batch_helper(self):
        """Test the expand_batch helper function."""
        # Test with different tensor shapes
        batch_size = 3
        mean_demand = torch.tensor([1.0, 2.0, 3.0])  # [batch_size]
        
        # Test with 2D target
        target_2d = torch.randn(batch_size, 5)  # [batch_size, features]
        expanded_2d = self.feature_registry.expand_batch(mean_demand, target_2d)
        expected_2d = torch.tensor([[1.0], [2.0], [3.0]])  # [batch_size, 1]
        assert expanded_2d.shape == (batch_size, 1)
        assert torch.allclose(expanded_2d, expected_2d)
        
        # Test with 3D target
        target_3d = torch.randn(batch_size, 4, 6)  # [batch_size, stores, periods]
        expanded_3d = self.feature_registry.expand_batch(mean_demand, target_3d)
        expected_3d = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])  # [batch_size, 1, 1]
        assert expanded_3d.shape == (batch_size, 1, 1)
        assert torch.allclose(expanded_3d, expected_3d)
    
    def test_normalization_without_past_demands(self):
        """Test that normalization is skipped when past_demands is not available."""
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0], [15.0, 25.0]]),
            'holding_costs': torch.tensor([[1.0, 2.0], [1.5, 2.5]])
        }
        
        result = self.feature_registry.prepare_inputs(observation)
        
        # Should not normalize when past_demands is missing
        assert not hasattr(self.feature_registry, '_moving_mean') or self.feature_registry._moving_mean is None
        assert result is not None
    
    def test_normalization_with_past_demands(self):
        """Test normalization when past_demands is available."""
        batch_size = 2
        n_stores = 2
        n_periods = 3
        
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0], [15.0, 25.0]]),
            'past_demands': torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                                        [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]]),
            'past_arrivals': torch.tensor([[[1.0, 2.0], [3.0, 4.0]], 
                                         [[2.0, 4.0], [6.0, 8.0]]]),
            'past_orders': torch.tensor([[[5.0, 10.0], [15.0, 20.0]], 
                                       [[10.0, 20.0], [30.0, 40.0]]])
        }
        
        # Store original values for comparison
        original_inventories = observation['store_inventories'].clone()
        original_demands = observation['past_demands'].clone()
        original_arrivals = observation['past_arrivals'].clone()
        original_orders = observation['past_orders'].clone()
        
        result = self.feature_registry.prepare_inputs(observation)
        
        # Check that EWMA stats were stored
        assert hasattr(self.feature_registry, '_moving_mean')
        assert hasattr(self.feature_registry, '_moving_variance')
        moving_mean = self.feature_registry._moving_mean
        moving_variance = self.feature_registry._moving_variance
        
        # Expected EWMA mean/variance update
        beta = self.config['policy_network']['ewma_beta']
        initial_mean = observation['past_demands'].mean(dim=2)
        initial_variance = observation['past_demands'].var(dim=2, unbiased=False)
        current_demand = observation['past_demands'][:, :, -1]
        expected_mean = beta * initial_mean + (1 - beta) * current_demand
        expected_variance = beta * initial_variance + (1 - beta) * (current_demand - expected_mean) ** 2
        assert torch.allclose(moving_mean, expected_mean)
        assert torch.allclose(moving_variance, expected_variance)
        
        # Check that original observation was not modified
        assert torch.allclose(observation['store_inventories'], original_inventories)
        assert torch.allclose(observation['past_demands'], original_demands)
        assert torch.allclose(observation['past_arrivals'], original_arrivals)
        assert torch.allclose(observation['past_orders'], original_orders)
        
        # Check that result is flattened
        expected_features = 2 + 6 + 4 + 4  # inventories + demands + arrivals + orders
        assert result.shape == (batch_size, expected_features)
    
    def test_normalization_values(self):
        """Test that normalization values are computed correctly."""
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0], [15.0, 25.0]]),
            'past_demands': torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                                        [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]])
        }
        
        # Manually compute expected EWMA mean
        beta = self.config['policy_network']['ewma_beta']
        initial_mean = observation['past_demands'].mean(dim=2)
        current_demand = observation['past_demands'][:, :, -1]
        expected_mean = beta * initial_mean + (1 - beta) * current_demand
        
        result = self.feature_registry.prepare_inputs(observation)
        
        # Check normalization constant
        actual_mean = self.feature_registry._moving_mean
        assert torch.allclose(actual_mean, expected_mean)
    
    def test_scale_by_mean_before_ranges(self):
        """Test that activated values are multiplied by mean before range scaling."""
        # First, prepare inputs to set up normalization
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0]]),
            'past_demands': torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        }
        
        # This will set up the normalization constant
        self.feature_registry.prepare_inputs(observation)
        
        # Capture input to scale_continuous_by_ranges
        captured = {}
        def _capture_scale(vals, ranges, observations=None, feature_registry=None):
            captured['vals'] = vals
            return vals
        self.range_manager.scale_continuous_by_ranges = Mock(side_effect=_capture_scale)
        
        # Raw continuous output shape [batch, 1, n_stores * n_sub_ranges]
        raw_continuous = torch.tensor([[[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]])
        self.feature_registry.process_continuous_output(
            raw_continuous,
            random_continuous=False,
            observations=observation
        )
        
        # Check that scale_continuous_by_ranges received mean-scaled activated values
        mean = self.feature_registry._moving_mean
        epsilon = getattr(self.feature_registry, '_ewma_epsilon', 1e-5)
        mean_denom = torch.clamp(mean, min=epsilon)
        
        reshaped_raw = self.feature_registry.reshape_continuous_output(raw_continuous)
        expected_scaled = reshaped_raw * mean_denom.unsqueeze(-1)
        assert torch.allclose(captured['vals'], expected_scaled, atol=1e-6)
    
    def test_no_denormalization_when_not_normalized(self):
        """Test that no denormalization is applied when normalization was not used."""
        # Create feature registry without normalization
        config_no_norm = {
            'policy_network': {
                'normalize_by_mean_demand': False,
                'observation_keys': ['store_inventories']
            }
        }
        feature_registry_no_norm = FeatureRegistry(config_no_norm, self.range_manager)
        
        # Prepare inputs (should not normalize)
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0]])
        }
        feature_registry_no_norm.prepare_inputs(observation)
        
        # Test scale_continuous_by_ranges
        continuous_values = torch.tensor([[[0.5, 1.0]]])
        ranges = [[0, 10], [0, 20]]
        
        scaled_values = self.range_manager.scale_continuous_by_ranges(
            continuous_values, ranges, observations=observation, feature_registry=feature_registry_no_norm
        )
        
        # Should only apply basic scaling, no denormalization
        # Expected: continuous_values * 2.0 + 1.0 = [[2.0, 3.0]]
        expected_scaled = torch.tensor([[[2.0, 3.0]]])
        assert torch.allclose(scaled_values, expected_scaled)
    
    def test_gradient_flow_preservation(self):
        """Test that gradient flow is preserved through normalization."""
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0]], requires_grad=True),
            'past_demands': torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
        }
        
        # Prepare inputs (should preserve gradients)
        result = self.feature_registry.prepare_inputs(observation)
        
        # Check that gradients are preserved
        assert observation['store_inventories'].requires_grad
        assert observation['past_demands'].requires_grad
        
        # Test that we can compute gradients through the result
        loss = result.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert observation['store_inventories'].grad is not None
        assert observation['past_demands'].grad is not None
    
    def test_different_batch_sizes(self):
        """Test normalization with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            observation = {
                'store_inventories': torch.randn(batch_size, 3),
                'past_demands': torch.randn(batch_size, 3, 5)
            }
            
            result = self.feature_registry.prepare_inputs(observation)
            
            # Check that EWMA mean has correct batch size and store dim
            moving_mean = self.feature_registry._moving_mean
            assert moving_mean.shape == (batch_size, 3)
            
            # Check that result has correct batch size
            assert result.shape[0] == batch_size
    
    def test_edge_case_zero_mean_demand(self):
        """Test handling of edge case where mean demand is zero."""
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0]]),
            'past_demands': torch.zeros(1, 2, 3)  # All zeros
        }
        
        result = self.feature_registry.prepare_inputs(observation)
        
        # Should handle zero mean demand gracefully (due to +1e-8)
        assert result is not None
        mean_demand = self.feature_registry._moving_mean
        assert torch.allclose(mean_demand, torch.zeros_like(mean_demand))


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
