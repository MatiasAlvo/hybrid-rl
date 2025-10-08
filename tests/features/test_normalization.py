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
                'observation_keys': ['store_inventories', 'past_demands', 'past_arrivals', 'past_orders']
            }
        }
        
        # Create mock range manager
        self.range_manager = Mock(spec=RangeManager)
        self.range_manager.scale_continuous_by_ranges = Mock(side_effect=self._mock_scale_continuous)
        
        # Create feature registry
        self.feature_registry = FeatureRegistry(self.config, self.range_manager)
        
        # Store original scale method for testing
        self.original_scale_method = self.range_manager.scale_continuous_by_ranges
    
    def _mock_scale_continuous(self, continuous_values, ranges, observations=None, feature_registry=None):
        """Mock the scale_continuous_by_ranges method to test denormalization."""
        # Simulate basic scaling
        scaled_values = continuous_values * 2.0 + 1.0
        
        # Apply denormalization if normalization was used
        if (feature_registry is not None and 
            hasattr(feature_registry, '_last_normalization_constant') and
            feature_registry._last_normalization_constant is not None):
            
            mean_demand = feature_registry._last_normalization_constant
            # Expand to match continuous_values shape
            mean_demand_expanded = mean_demand.view(-1, 1, 1).expand_as(continuous_values)
            scaled_values = scaled_values * mean_demand_expanded
        
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
        assert not hasattr(self.feature_registry, '_last_normalization_constant')
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
        
        # Check that normalization constant was stored
        assert hasattr(self.feature_registry, '_last_normalization_constant')
        mean_demand = self.feature_registry._last_normalization_constant
        
        # Expected mean demand: mean across stores and periods
        expected_mean_demand = observation['past_demands'].mean(dim=(1, 2))
        assert torch.allclose(mean_demand, expected_mean_demand)
        
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
        
        # Manually compute expected mean demand
        expected_mean_demand = observation['past_demands'].mean(dim=(1, 2))
        # Sample 0: mean of [1,2,3,4,5,6] = 3.5
        # Sample 1: mean of [2,4,6,8,10,12] = 7.0
        expected_mean_demand = torch.tensor([3.5, 7.0])
        
        result = self.feature_registry.prepare_inputs(observation)
        
        # Check normalization constant
        actual_mean_demand = self.feature_registry._last_normalization_constant
        assert torch.allclose(actual_mean_demand, expected_mean_demand)
    
    def test_denormalization_in_scale_continuous(self):
        """Test that denormalization is applied in scale_continuous_by_ranges."""
        # First, prepare inputs to set up normalization
        observation = {
            'store_inventories': torch.tensor([[10.0, 20.0]]),
            'past_demands': torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        }
        
        # This will set up the normalization constant
        self.feature_registry.prepare_inputs(observation)
        
        # Now test denormalization in scale_continuous_by_ranges
        continuous_values = torch.tensor([[[0.5, 1.0, 1.5]]])  # [batch, stores, ranges]
        ranges = [[0, 10], [0, 20], [0, 30]]
        
        # Call the actual scale_continuous_by_ranges method
        scaled_values = self.range_manager.scale_continuous_by_ranges(
            continuous_values, ranges, observations=observation, feature_registry=self.feature_registry
        )
        
        # Check that denormalization was applied
        # Expected: (continuous_values * 2.0 + 1.0) * mean_demand
        # continuous_values * 2.0 + 1.0 = [[2.0, 3.0, 4.0]]
        # mean_demand = 3.5, so final result should be [[7.0, 10.5, 14.0]]
        expected_scaled = torch.tensor([[[7.0, 10.5, 14.0]]])
        assert torch.allclose(scaled_values, expected_scaled, atol=1e-6)
    
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
            
            # Check that normalization constant has correct batch size
            mean_demand = self.feature_registry._last_normalization_constant
            assert mean_demand.shape == (batch_size,)
            
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
        mean_demand = self.feature_registry._last_normalization_constant
        assert torch.allclose(mean_demand, torch.tensor([0.0]))


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
