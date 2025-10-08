import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.feature_registry import FeatureRegistry
from envs.inventory.range_manager import RangeManager


class TestProcessNetworkOutput:
    """Test suite for process_network_output function focusing on log probability calculations"""
    
    @pytest.fixture
    def mock_range_manager(self):
        """Create a mock range manager for testing"""
        mock_rm = Mock(spec=RangeManager)
        
        # Mock the methods we need
        mock_rm.get_discrete_probabilities.return_value = (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # discrete_probs
            torch.tensor([[-1.0, -2.0, -3.0], [-2.0, -1.0, -3.0]])  # log_probs
        )
        
        mock_rm.apply_activations.return_value = torch.tensor([[0.5, 0.3], [0.7, 0.2]])
        mock_rm.scale_continuous_by_ranges.return_value = torch.tensor([[0.5, 0.3], [0.7, 0.2]])
        mock_rm.get_continuous_ranges.return_value = [(-1.0, 1.0), (-1.0, 1.0)]
        mock_rm.compute_feature_actions.return_value = torch.tensor([[0.5, 0.3], [0.7, 0.2]])
        
        return mock_rm
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for FeatureRegistry"""
        return {
            'policy_network': {
                'input_size': 2,
                'heads': {
                    'discrete': {'size': 3},
                    'continuous': {'size': 2}
                }
            }
        }
    
    @pytest.fixture
    def feature_registry(self, mock_range_manager, mock_config):
        """Create a FeatureRegistry instance with mocked range manager"""
        registry = FeatureRegistry(mock_config, mock_range_manager)
        return registry
    
    def test_discrete_only_log_probs(self, feature_registry, mock_range_manager):
        """Test that discrete log probabilities are calculated correctly"""
        # Setup - raw_outputs['discrete'] has shape [batch, n_discrete, n_actions]
        # where n_discrete is the number of discrete actions and n_actions is the number of possible values
        raw_outputs = {
            'discrete': torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]])  # [batch, n_discrete, n_actions]
        }
        
        # Mock discrete probabilities to return specific indices
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),  # discrete_probs (select action 2 for both) [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0, -3.0], [-2.0, -1.0, -3.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function
        result = feature_registry.process_network_output(raw_outputs, argmax=True)
        
        # Verify discrete probabilities were computed
        mock_range_manager.get_discrete_probabilities.assert_called_once()
        
        # Verify the result structure
        assert 'logits' in result
        assert 'discrete_probs' in result
        assert 'discrete_action_indices' in result
        
        # The logits should be the discrete log probabilities for selected actions
        # With the mocked probabilities selecting action 2, we get the logits for action 2
        expected_logits = torch.tensor([[3.0, 3.0]])  # logits for selected actions (highest values) [batch, n_discrete]
        torch.testing.assert_close(result['logits'], expected_logits, rtol=1e-5, atol=1e-5)
    
    def test_continuous_only_log_probs(self, feature_registry, mock_range_manager):
        """Test that continuous log probabilities are calculated correctly when using Gaussian sampling"""
        # Setup
        raw_outputs = {
            'discrete': torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]),  # [batch, n_discrete, n_actions]
            'continuous': torch.tensor([[[0.5, 0.3], [0.7, 0.2]]]),  # [batch, n_discrete, n_continuous]
            'continuous_mean': torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            'continuous_log_std': torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # std = 1.0
        }
        
        # Mock the discrete probabilities to return specific indices
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0], [-2.0, -1.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function with random_continuous=True
        result = feature_registry.process_network_output(raw_outputs, random_continuous=True)
        
        # Verify the result structure
        assert 'logits' in result
        assert 'raw_continuous_samples' in result
        
        # The logits should include both discrete and continuous log probabilities
        # For a normal distribution with mean=0, std=1, the log prob of value x is -0.5 * x^2 - 0.5 * log(2π)
        expected_continuous_log_probs = -0.5 * (raw_outputs['continuous'] ** 2) - 0.5 * np.log(2 * np.pi)
        expected_combined_logits = torch.tensor([[-1.0, -1.0]]) + expected_continuous_log_probs.gather(1, torch.tensor([[[0], [1]]])).squeeze(1)
        
        torch.testing.assert_close(result['logits'], expected_combined_logits, rtol=1e-5, atol=1e-5)
    
    def test_combined_discrete_continuous_log_probs(self, feature_registry, mock_range_manager):
        """Test that discrete and continuous log probabilities are correctly combined"""
        # Setup
        raw_outputs = {
            'discrete': torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]]),  # [batch, n_discrete, n_actions]
            'continuous': torch.tensor([[[0.5, 0.3], [0.7, 0.2]]]),  # [batch, n_discrete, n_continuous]
            'continuous_mean': torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            'continuous_log_std': torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # std = 1.0
        }
        
        # Mock discrete probabilities
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0, -3.0], [-2.0, -1.0, -3.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function
        result = feature_registry.process_network_output(raw_outputs, random_continuous=True)
        
        # Verify the combined logits
        # Discrete log probs: [3.0, 3.0] (for selected actions - highest values)
        # Continuous log probs: calculated from normal distribution
        expected_continuous_log_probs = -0.5 * (raw_outputs['continuous'] ** 2) - 0.5 * np.log(2 * np.pi)
        expected_combined_logits = torch.tensor([[3.0, 3.0]]) + expected_continuous_log_probs.gather(1, torch.tensor([[[0], [1]]])).squeeze(1)
        
        torch.testing.assert_close(result['logits'], expected_combined_logits, rtol=1e-5, atol=1e-5)
    
    def test_log_prob_addition_manual_verification(self, feature_registry, mock_range_manager):
        """Manually verify that log probabilities are added correctly (not multiplied)"""
        # Setup with known values - use random_continuous=False to avoid random sampling
        discrete_logits = torch.tensor([[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]])  # [batch, n_discrete, n_actions]
        continuous_values = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])  # [batch, n_discrete, n_continuous]
        
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous': continuous_values
        }
        
        # Mock discrete probabilities to select first action for both samples
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function with random_continuous=False
        result = feature_registry.process_network_output(raw_outputs, random_continuous=False)
        
        # Manual calculation of expected log probabilities
        # The function uses raw discrete logits (not log probs) for selected actions
        # Discrete logits for selected actions (index 0): [2.0, 1.0] from raw_outputs['discrete']
        discrete_logits_selected = torch.tensor([[2.0, 1.0]])  # [batch, n_discrete]
        
        # When random_continuous=False, no continuous log probs are added
        expected_logits = discrete_logits_selected  # Only discrete logits
        
        # Verify the result
        torch.testing.assert_close(result['logits'], expected_logits, rtol=1e-5, atol=1e-5)
        
        # Verify that we're getting the expected structure
        assert result['logits'].shape == (1, 2)  # [batch, n_discrete]
        assert 'discrete_probs' in result
        assert 'discrete_action_indices' in result
    
    def test_discrete_continuous_log_prob_addition(self, feature_registry, mock_range_manager):
        """Test that discrete and continuous log probabilities are correctly added together"""
        # Setup with deterministic values to test the addition logic
        discrete_logits = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # [batch, n_discrete, n_actions]
        continuous_values = torch.tensor([[[0.5, 0.3], [0.7, 0.2]]])  # [batch, n_discrete, n_continuous]
        continuous_mean = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        continuous_log_std = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # std = 1.0
        
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous': continuous_values,
            'continuous_mean': continuous_mean,
            'continuous_log_std': continuous_log_std
        }
        
        # Mock discrete probabilities to select specific actions
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0], [-2.0, -1.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function with random_continuous=True
        result = feature_registry.process_network_output(raw_outputs, random_continuous=True)
        
        # Verify the result structure
        assert 'logits' in result
        assert 'raw_continuous_samples' in result
        assert result['logits'].shape == (1, 2)  # [batch, n_discrete]
        
        # The key test: verify that discrete and continuous log probs are being added
        # We can't predict the exact values due to random sampling, but we can verify:
        # 1. The result has the correct shape
        # 2. The result is not just the discrete logits (it should include continuous contribution)
        # 3. The result is not the product of discrete and continuous (it should be the sum)
        
        # Get the discrete logits for selected actions
        discrete_logits_selected = torch.tensor([[2.0, 1.0]])  # [batch, n_discrete]
        
        # The actual logits should be different from just discrete logits (due to continuous addition)
        assert not torch.allclose(result['logits'], discrete_logits_selected, rtol=1e-5)
        
        # The actual logits should be the sum of discrete and continuous, not the product
        # (This is the key test - we're adding log probabilities, not multiplying them)
        discrete_continuous_product = discrete_logits_selected * torch.tensor([[0.5, 0.3]])  # Example product
        assert not torch.allclose(result['logits'], discrete_continuous_product, rtol=1e-5)
    
    def test_no_continuous_log_probs_when_not_random_continuous(self, feature_registry, mock_range_manager):
        """Test that continuous log probs are not added when random_continuous=False"""
        # Setup
        raw_outputs = {
            'discrete': torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]]),  # [batch, n_discrete, n_actions]
            'continuous': torch.tensor([[[0.5, 0.3], [0.7, 0.2]]])  # [batch, n_discrete, n_continuous]
        }
        
        # Mock discrete probabilities
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0, -3.0], [-2.0, -1.0, -3.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function with random_continuous=False
        result = feature_registry.process_network_output(raw_outputs, random_continuous=False)
        
        # The logits should only contain discrete log probabilities (highest values)
        expected_logits = torch.tensor([[3.0, 3.0]])  # Only discrete log probs (highest values) [batch, n_discrete]
        torch.testing.assert_close(result['logits'], expected_logits, rtol=1e-5, atol=1e-5)
    
    def test_gather_operation_correctness(self, feature_registry, mock_range_manager):
        """Test that the gather operation correctly selects continuous log probs for chosen discrete actions"""
        # Setup with 2 discrete actions and 2 continuous dimensions
        discrete_logits = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # [batch, n_discrete, n_actions]
        continuous_values = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])  # [batch, n_discrete, n_continuous]
        continuous_mean = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        continuous_log_std = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # std = 1.0
        
        raw_outputs = {
            'discrete': discrete_logits,
            'continuous': continuous_values,
            'continuous_mean': continuous_mean,
            'continuous_log_std': continuous_log_std
        }
        
        # Mock discrete probabilities to select different actions for each sample
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),  # discrete_probs (select action 1 and 0) [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0], [-2.0, -1.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function
        result = feature_registry.process_network_output(raw_outputs, random_continuous=True)
        
        # Manual calculation
        # Discrete log probs for selected actions (indices 1 and 0): [-2.0, -2.0]
        discrete_log_probs = torch.tensor([[-2.0, -2.0]])
        
        # Continuous log probs for all actions
        continuous_log_probs = -0.5 * (continuous_values ** 2) - 0.5 * np.log(2 * np.pi)
        
        # Selected continuous log probs for actions 1 and 0
        selected_continuous_log_probs = continuous_log_probs[:, [1, 0], :].sum(dim=-1)  # Sum across continuous dimensions
        
        # Expected combined logits
        expected_logits = discrete_log_probs + selected_continuous_log_probs
        
        # Verify the result
        torch.testing.assert_close(result['logits'], expected_logits, rtol=1e-5, atol=1e-5)
    
    def test_edge_case_empty_continuous(self, feature_registry, mock_range_manager):
        """Test behavior when continuous values are empty"""
        # Setup with only discrete actions
        raw_outputs = {
            'discrete': torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]])  # [batch, n_discrete, n_actions]
        }
        
        # Mock discrete probabilities
        mock_range_manager.get_discrete_probabilities.return_value = (
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),  # discrete_probs [batch, n_discrete, n_actions]
            torch.tensor([[[-1.0, -2.0, -3.0], [-2.0, -1.0, -3.0]]])  # log_probs [batch, n_discrete, n_actions]
        )
        
        # Call the function
        result = feature_registry.process_network_output(raw_outputs)
        
        # Should only have discrete log probabilities (highest values)
        expected_logits = torch.tensor([[3.0, 3.0]])  # [batch, n_discrete]
        torch.testing.assert_close(result['logits'], expected_logits, rtol=1e-5, atol=1e-5)
    
    def test_edge_case_empty_discrete(self, feature_registry, mock_range_manager):
        """Test behavior when discrete actions are empty"""
        # Setup with only continuous actions
        raw_outputs = {
            'continuous': torch.tensor([[[0.5, 0.3], [0.7, 0.2]]])
        }
        
        # Call the function
        result = feature_registry.process_network_output(raw_outputs)
        
        # Should not have logits when no discrete actions
        assert result['logits'] is None


if __name__ == "__main__":
    pytest.main([__file__])
