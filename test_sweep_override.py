#!/usr/bin/env python3
"""Test script to verify discrete_lr_multiplier override logic"""

# Simulate the sweep.py override logic
def test_override():
    # Simulate hyperparams_config
    hyperparams_config = {
        'optimizer_params': {
            'learning_rate': 0.005,
            'lr_multipliers': {
                'value': 1.0,
                'backbone': 1.0,
                'continuous': 1.0,
                'discrete': 1.0,
                'other': 1.0
            }
        }
    }
    
    # Simulate sweep parameter
    param_name = 'discrete_lr_multiplier'
    param_value = 0.1  # Sampled from sweep
    
    # Mapping from sweep.py line 130
    param_path = ['optimizer_params', 'lr_multipliers', 'discrete']
    
    # Navigate to the correct nested dict (sweep.py lines 143-149)
    current = hyperparams_config
    for i, key in enumerate(param_path[:-1]):
        if isinstance(current, dict):
            if key not in current:
                current[key] = {} if i < len(param_path) - 2 else []
            current = current[key]
    
    # Set the final value (sweep.py lines 160-161)
    if isinstance(current, dict):
        current[param_path[-1]] = param_value
    
    # Verify the override worked
    print("Testing discrete_lr_multiplier override:")
    print(f"  Input value: {param_value}")
    print(f"  Result: {hyperparams_config['optimizer_params']['lr_multipliers']['discrete']}")
    print(f"  Full lr_multipliers: {hyperparams_config['optimizer_params']['lr_multipliers']}")
    
    # Check that it matches
    assert hyperparams_config['optimizer_params']['lr_multipliers']['discrete'] == param_value
    print("\n✓ Override logic works correctly!")
    
    # Simulate how it would be used in main_run.py
    lr_multipliers = hyperparams_config['optimizer_params']['lr_multipliers']
    base_lr = hyperparams_config['optimizer_params']['learning_rate']
    discrete_multiplier = lr_multipliers.get('discrete', 1.0)
    final_discrete_lr = base_lr * discrete_multiplier
    
    print(f"\nSimulated usage in main_run.py:")
    print(f"  Base LR: {base_lr}")
    print(f"  Discrete multiplier: {discrete_multiplier}")
    print(f"  Final discrete LR: {final_discrete_lr}")
    
    expected_lr = 0.005 * 0.1
    assert abs(final_discrete_lr - expected_lr) < 1e-9
    print(f"\n✓ Final learning rate calculation is correct! ({expected_lr})")

if __name__ == "__main__":
    test_override()

