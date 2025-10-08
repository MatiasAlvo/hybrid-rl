# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.envs.inventory.range_manager import RangeManager

def test_range_manager():
    # Create a simplified config similar to your hybrid_general.yml
    config = {
        'discrete_features': {
            'fixed_costs': {
                'thresholds': [0, 50, 100, 200],
                'values': [1, 5, 10]
            },
            'bulk_discounts': {
                'thresholds': [0, 50, 70, 200],
                'values': [0.0, 0.1, 0.2]
            }
        }
    }

    # Initialize RangeManager
    manager = RangeManager(config, device='cpu')

    # Print computed ranges
    print("\nComputed Ranges:")
    for i, range_ in enumerate(manager.ranges):
        print(f"Range {i}: [{range_[0]}, {range_[1]}]")

    # Create sample network output (batch size of 2)
    batch_size = 2
    n_ranges = len(manager.ranges)
    
    # Create sample discrete probabilities
    discrete_probs = torch.zeros((batch_size, n_ranges))
    discrete_probs[0] = torch.tensor([0.4, 0.3, 0.2, 0.1])  # First sample
    discrete_probs[1] = torch.tensor([0.1, 0.4, 0.4, 0.1])  # Second sample
    
    # Create sample continuous values (between 0 and 1)
    continuous_values = torch.tensor([
        [0.2, 0.5, 0.7, 0.9],  # First sample
        [0.3, 0.6, 0.4, 0.8]   # Second sample
    ])

    # add dim=1 to both continuous_values and discrete_probs
    continuous_values = continuous_values.unsqueeze(1)
    discrete_probs = discrete_probs.unsqueeze(1)

    # Convert network output
    result = manager.convert_network_output_to_simulator_action(
        discrete_probs,
        continuous_values,
        use_argmax=False
    )

    # Print results
    print("\nNetwork Output Conversion Results:")
    print("\nDiscrete Probabilities:")
    print(discrete_probs)

    #print original continuous values
    print("\nOriginal Continuous Values:")
    print(continuous_values)
    
    print("\nContinuous Values (scaled to ranges):")
    print(result['continuous_per_range'])
    
    print("\nFeature Mappings:")
    for feature_name, mapping in result['feature_mappings'].items():
        print(f"\n{feature_name}:")
        print(f"Discrete probabilities: {mapping['discrete']}")
        print(f"Continuous values: {mapping['continuous']}")
        print(f"Feature values: {mapping['value']}")

if __name__ == "__main__":
    test_range_manager() 