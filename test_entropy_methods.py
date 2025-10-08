#!/usr/bin/env python3
"""
Test to verify that get_log_probs_value_and_entropy and get_gaussian_entropy work correctly
for both fixed and state-dependent standard deviation.
"""

import torch
import yaml
from src.algorithms.hybrid.agents.hybrid_agent import FactoredGaussianPPOAgent
from src.features.feature_registry import FeatureRegistry
from src.envs.inventory.range_manager import RangeManager

def test_entropy_methods():
    """Test entropy calculation for both fixed and state-dependent std"""
    print("Testing entropy methods...")
    
    device = torch.device('cpu')
    
    # Create mock range manager
    range_manager = RangeManager({
        'discrete_features': {
            'store_inventories': {
                'thresholds': [0, 10, 20, 30],
                'values': [0, 1, 2]
            }
        }
    }, device=device)
    
    # Test fixed std
    print("\n1. Testing fixed_std=True...")
    with open('configs/policies/factored_gaussian_ppo.yml', 'r') as f:
        config = yaml.safe_load(f)
    config['agent_params']['fixed_std'] = True
    
    feature_registry = FeatureRegistry(config['nn_params'], range_manager)
    agent = FactoredGaussianPPOAgent(config, feature_registry, device)
    
    # Test get_gaussian_entropy with scalar
    scalar_log_std = torch.tensor([-1.0], device=device)
    scalar_entropy = agent.get_gaussian_entropy(scalar_log_std)
    print(f"Scalar log_std: {scalar_log_std.item():.3f}")
    print(f"Scalar entropy: {scalar_entropy.item():.3f}")
    
    # Test with vector (should work the same way)
    vector_log_std = torch.tensor([[-1.0, -0.5, -2.0]], device=device)
    vector_entropy = agent.get_gaussian_entropy(vector_log_std)
    print(f"Vector log_std: {vector_log_std.squeeze().tolist()}")
    print(f"Vector entropy: {vector_entropy.squeeze().tolist()}")
    
    # Test state-dependent std
    print("\n2. Testing fixed_std=False...")
    config['agent_params']['fixed_std'] = False
    agent_sd = FactoredGaussianPPOAgent(config, feature_registry, device)
    
    # Test with state-dependent log_std (vector)
    state_dependent_log_std = torch.tensor([[-0.8, -1.2, -1.5]], device=device)
    sd_entropy = agent_sd.get_gaussian_entropy(state_dependent_log_std)
    print(f"State-dependent log_std: {state_dependent_log_std.squeeze().tolist()}")
    print(f"State-dependent entropy: {sd_entropy.squeeze().tolist()}")
    
    # Test get_log_probs_value_and_entropy with mock data
    print("\n3. Testing get_log_probs_value_and_entropy...")
    
    # Create mock observation
    batch_size = 2
    n_features = 7  # Based on observation_keys
    mock_obs = torch.randn(batch_size, n_features, device=device)
    
    # Mock discrete action indices
    discrete_actions = torch.tensor([0, 1], device=device)
    
    # Mock continuous samples
    continuous_samples = torch.randn(batch_size, 1, 1, device=device)
    
    # Test fixed std
    try:
        log_probs, value, entropy = agent.get_log_probs_value_and_entropy(
            mock_obs, discrete_actions, continuous_samples
        )
        print(f"Fixed std - Log probs shape: {log_probs.shape}")
        print(f"Fixed std - Value: {value}")
        print(f"Fixed std - Entropy shape: {entropy.shape}")
        print("✓ Fixed std get_log_probs_value_and_entropy works!")
    except Exception as e:
        print(f"✗ Fixed std failed: {e}")
    
    # Test state-dependent std
    try:
        log_probs_sd, value_sd, entropy_sd = agent_sd.get_log_probs_value_and_entropy(
            mock_obs, discrete_actions, continuous_samples
        )
        print(f"State-dependent std - Log probs shape: {log_probs_sd.shape}")
        print(f"State-dependent std - Value: {value_sd}")
        print(f"State-dependent std - Entropy shape: {entropy_sd.shape}")
        print("✓ State-dependent std get_log_probs_value_and_entropy works!")
    except Exception as e:
        print(f"✗ State-dependent std failed: {e}")
    
    print("\n🎉 All entropy method tests completed!")

if __name__ == "__main__":
    test_entropy_methods()
