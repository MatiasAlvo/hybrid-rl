from src.envs.inventory.simulator import Simulator
from src import torch
from typing import Dict, Optional, Tuple
from src.envs.inventory.range_manager import RangeManager

class HybridSimulator(Simulator):
    """
    Simulator for hybrid action spaces in inventory control.
    Supports various cost structures and transition dynamics through modular components.
    """
    def __init__(self, feature_registry, device='cpu'):
        super().__init__(device)
        self.feature_registry = feature_registry
        self.action_ranges = self.feature_registry.get_simulator_config()
        
        # Initialize dictionaries for cost and transition functions
        self.cost_functions = {}
        self.transition_functions = {}
        
    def reset(self, periods, problem_params, data, observation_params):
        """Initialize problem-specific cost and transition functions"""
        # Register cost functions based on problem parameters
        self._register_cost_functions(problem_params['discrete_features'])
        
        return super().reset(periods, problem_params, data, observation_params)
    
    def step(self, observation, action_dict):
        """Execute one simulation step"""
        # First handle basic transitions and costs (similar to base simulator)
        next_state, base_costs = self._calculate_base_transitions_and_costs(observation, action_dict)
        
        # Then add additional costs from registered components
        additional_costs = self._calculate_additional_costs(observation, action_dict)
        total_costs = base_costs + additional_costs
        
        return next_state, total_costs, False, {}, {}
        
    def _calculate_base_transitions_and_costs(self, observation, action_dict):
        """Calculate basic transitions and costs with proper in-place updates"""

        # 1. Get current demands
        current_demands = self.get_current_demands(
            self._internal_data,
            observation['current_period'].item()
        )
        
        # 2. Calculate post-demand inventory
        inventory = observation['store_inventories']
        inventory_on_hand = inventory[:, :, 0]
        post_inventory_on_hand = inventory_on_hand - current_demands
        
        # 3. Calculate variable costs following base simulator pattern
        if self.maximize_profit:
            base_costs = (
                -observation['underage_costs'] * torch.minimum(inventory_on_hand, current_demands) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
            )
        else:
            base_costs = (
                observation['underage_costs'] * torch.clip(-post_inventory_on_hand, min=0) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
            )
        
        # Add procurement costs
        base_costs += observation['procurement_costs']*action_dict['feature_actions']['total_action']
        
        # Sum costs across stores
        base_costs = base_costs.sum(dim=1)
        
        # 4. Handle lost demand if needed
        if self.problem_params.get('lost_demand', False):
            post_inventory_on_hand = torch.clip(post_inventory_on_hand, min=0)
        
        # 5. Get allocation from feature actions
        feature_actions = action_dict['feature_actions']
        order_feature = list(feature_actions.keys())[0]
        allocation = feature_actions[order_feature]['action'].sum(dim=-1)
        
        # 6. Update inventories using same pattern as base simulator
        lead_time = self.problem_params.get('lead_time', 1)
        observation['store_inventories'] = self._update_inventories_in_place(
            inventory,
            post_inventory_on_hand,
            allocation,
            lead_time
        )

        return observation, base_costs
    
    def _calculate_additional_costs(self, observation, action_dict):
        """Calculate additional costs from registered components"""
        additional_costs = torch.zeros(self.batch_size, device=self.device)
        
        # Apply all registered cost functions
        for cost_func in self.cost_functions.values():
            additional_costs += cost_func(observation, action_dict)
        
        return additional_costs
    
    def _update_orders(self, observation, action_dict):
        """Update orders based on lead time structure"""
        if self.problem_params.get('variable_lead_time'):
            self._update_variable_lead_time_orders(observation, action_dict)
        else:
            self._update_single_lead_time_orders(observation, action_dict)
    
    def _update_variable_lead_time_orders(self, observation, action_dict):
        """Update orders with variable lead times"""
        discrete_probs = action_dict['discrete_probs']
        continuous_values = action_dict['continuous_values']
        
        # For each lead time possibility
        for lt_idx, lead_time in enumerate(self.lead_time_values):
            # Get order amount for this lead time
            order_amount = discrete_probs[..., lt_idx] * continuous_values[..., lt_idx]
            
            # Add to appropriate future period
            if lead_time < observation['store_inventories'].shape[-1]:
                observation['store_inventories'][:, :, lead_time] += order_amount
    
    def _calculate_holding_costs(self, observation, post_demand_inventory):
        """Calculate holding costs"""
        holding_cost = torch.where(
            post_demand_inventory > 0,
            post_demand_inventory * self.problem_params['holding_cost'],
            torch.zeros_like(post_demand_inventory)
        )
        return holding_cost.sum(dim=1)
    
    def _calculate_shortage_costs(self, post_demand_inventory):
        """Calculate shortage costs"""
        shortage_cost = torch.where(
            post_demand_inventory < 0,
            -post_demand_inventory * self.problem_params['shortage_cost'],
            torch.zeros_like(post_demand_inventory)
        )
        return shortage_cost.sum(dim=1)
    
    def _register_cost_functions(self, problem_params):
        """Register cost functions based on problem configuration"""
        # Base costs (always included)
        # self.cost_functions['holding'] = self._calculate_holding_costs
        # self.cost_functions['shortage'] = self._calculate_shortage_costs
        
        # Optional costs based on problem parameters
        if problem_params.get('fixed_ordering_cost'):
            self.cost_functions['fixed_ordering'] = self._calculate_fixed_ordering_costs
        if problem_params.get('bulk_discounts'):
            self.cost_functions['bulk_discount'] = self._calculate_bulk_discount_costs
    
    def _update_current_inventory(self, observation, action):
        """Update current period inventory (without new orders)"""
        current_inventory = observation['store_inventories'][:, :, 0]
        current_demands = self.get_current_demands(
            self._internal_data,
            observation['current_period'].item()
        )
        
        return current_inventory - current_demands
    
    def _update_pending_orders(self, next_state, action):
        """Update pending orders for variable lead times"""
        discrete_probs = action['discrete_probs']
        continuous_values = action['continuous_values']
        
        # For each lead time possibility
        for lt_idx, lead_time in enumerate(self.lead_time_values):
            # Get order amount for this lead time
            order_amount = discrete_probs[..., lt_idx] * continuous_values[..., lt_idx]
            
            # Add to appropriate future period
            if lead_time < next_state['store_inventories'].shape[-1]:
                next_state['store_inventories'][:, :, lead_time] += order_amount
    
    # Cost calculation methods
    def _calculate_fixed_ordering_costs(self, observation, action_dict):
        """Calculate fixed ordering costs"""
        feature_probs = action_dict['feature_actions']['fixed_ordering_cost']['range_probs']
        feature_values = action_dict['feature_actions']['fixed_ordering_cost']['values']
        fixed_costs = (feature_probs * feature_values.unsqueeze(0)).sum(dim=(-1, -2))
        return fixed_costs
    
    def _calculate_bulk_discount_costs(self, observation, action_dict):
        """Calculate bulk discount costs"""
        # calculate average discount by multiplying the bulk discount feature action by its values
        bulk_discount_probs = action_dict['feature_actions']['bulk_discounts']['range_probs']
        bulk_discount_values = action_dict['feature_actions']['bulk_discounts']['values']
        average_discount = (bulk_discount_probs * bulk_discount_values.unsqueeze(0)).sum(dim=(-1))
        return -(average_discount*observation['procurement_costs']*action_dict['feature_actions']['total_action']).sum(dim=1)
    
    # Transition methods
    def _update_base_inventory(self, observation, action_dict):
        """Update base inventory levels"""
        # Implementation
        pass
    
    def _update_variable_lead_time(self, observation, action_dict):
        """Update lead times"""
        # Implementation
        pass
    
    def _apply_capacity_constraints(self, observation, action_dict):
        """Apply capacity constraints"""
        # Implementation
        pass
    
    def _update_single_lead_time_orders(self, next_state, action_dict):
        """Update orders with single lead time"""
        # Get total action for each store from feature_actions
        feature_actions = action_dict['feature_actions']
        
        # Assuming the first feature corresponds to order quantities
        order_feature = list(feature_actions.keys())[0]
        order_amounts = feature_actions[order_feature]['action'].sum(dim=-1)  # Sum over ranges
        
        # Add orders to the appropriate future period based on lead time
        lead_time = self.problem_params.get('lead_time', 1)
        if lead_time < next_state['store_inventories'].shape[-1]:
            next_state['store_inventories'][:, :, lead_time] += order_amounts
    
    def _update_inventories_in_place(self, inventory, inventory_on_hand, allocation, lead_time):
        """
        Update inventory in-place following same pattern as update_inventory_for_heterogeneous_lead_times
        
        Args:
            inventory: Current inventory tensor [batch_size, n_stores, max_lead_time]
            inventory_on_hand: Post-demand inventory [batch_size, n_stores]
            allocation: Order amounts [batch_size, n_stores]
            lead_time: Lead time for orders
        """
        return torch.stack(
            [
                inventory_on_hand + inventory[:, :, 1],  # Current + next period arrivals
                *self.move_columns_left(inventory, 1, lead_time - 1),  # Move existing orders left
                allocation  # Add new orders
            ], 
            dim=2
        )