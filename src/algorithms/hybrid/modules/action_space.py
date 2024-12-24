class HybridActionSpace:
    """Defines structure and transformations for hybrid action spaces"""
    def __init__(self, config):
        self.action_type = config.get('action_type', 'range_selection')  # or 'fixed_cost'
        
        if self.action_type == 'range_selection':
            # Each range is defined by [min, max]
            self.ranges = config['action_ranges']  # e.g., [[1,3], [5,8], [11,15]]
            self.n_ranges = len(self.ranges)
        
        elif self.action_type == 'fixed_cost':
            self.continuous_bounds = config['continuous_bounds']  # e.g., [0, 100]
            
    def transform_to_real_action(self, raw_action):
        """Transform network outputs to actual action values"""
        if self.action_type == 'range_selection':
            range_idx = raw_action['discrete']['range_selection']  # Index of selected range
            normalized_value = raw_action['continuous']['range_value']  # Value between 0 and 1
            
            # Get selected range bounds
            selected_range = self.ranges[range_idx]
            min_val, max_val = selected_range
            
            # Transform normalized value to actual value within range
            actual_value = min_val + (max_val - min_val) * normalized_value
            
            return {
                'selected_range': range_idx,
                'value': actual_value
            }
            
        elif self.action_type == 'fixed_cost':
            order_decision = raw_action['discrete']['order']  # Binary decision
            normalized_quantity = raw_action['continuous']['quantity']  # Between 0 and 1
            
            # Transform to actual quantity if ordering
            min_val, max_val = self.continuous_bounds
            actual_quantity = min_val + (max_val - min_val) * normalized_quantity
            
            return {
                'order': order_decision,
                'quantity': actual_quantity if order_decision else 0.0
            } 