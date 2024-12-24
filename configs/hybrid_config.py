config = {
    'action_space': {
        'action_type': 'fixed_cost',
        'continuous_bounds': [0, 100]
    },
    'cost_structure': 'fixed_plus_variable',
    'fixed_cost': 10.0,
    'variable_cost': 2.0,
    'training': {
        'discrete_loss_weight': 1.0,
        'continuous_loss_weight': 1.0
    }
} 