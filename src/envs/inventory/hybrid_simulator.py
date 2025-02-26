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
        
        # Initialize normalization parameters
        self.inventory_mean = 0.0
        self.inventory_std = 1.0
        self.normalize_observations = False
        
        # Store inventory observations during steps
        self._inventory_observations = []
        
    def reset(self, periods, problem_params, data, observation_params):
        """Initialize problem-specific cost and transition functions"""
        # Register cost functions based on problem parameters
        self._register_cost_functions(problem_params['discrete_features'])
        
        # Get normalize_observations from observation_params
        self.normalize_observations = observation_params.get('normalize_observations', False)
        
        # Compute normalization stats if we have collected data
        if self.normalize_observations and len(self._inventory_observations) > 0:
            self._compute_normalization_stats()
        
        # Clear the observations list for next batch
        self._inventory_observations = []
        
        # Get base observation (unnormalized)
        observation, info = super().reset(periods, problem_params, data, observation_params)
        
        # Store unnormalized observation internally
        self._internal_observation = observation.copy()
        
        # Add initial observation to our collection
        if self.normalize_observations:
            self._inventory_observations.append(observation['store_inventories'][..., 0])
            observation = self._normalize_observation(observation)
        
        return observation, info
    
    def step(self, observation, action_dict):
        """Execute one simulation step"""
        # Use internal unnormalized observation for calculations
        next_observation, base_costs = self._calculate_base_transitions_and_costs(self._internal_observation, action_dict)
        
        # Update internal observation
        self._internal_observation = next_observation.copy()
        
        # Store inventory on hand for normalization
        if self.normalize_observations:
            self._inventory_observations.append(next_observation['store_inventories'][..., 0].detach())
        
        # Then add additional costs from registered components
        additional_costs = self._calculate_additional_costs(self._internal_observation, action_dict)
        total_costs = base_costs + additional_costs

        # Return normalized observation if needed
        if self.normalize_observations:
            next_observation = self._normalize_observation(next_observation)

        return next_observation, total_costs, False, {}, {}
        
    def _calculate_base_transitions_and_costs(self, observation, action_dict):
        """Calculate basic transitions and costs with proper in-place updates"""

        # 1. Get current demands
        current_demands = self.get_current_demands(
            self._internal_data,
            observation['current_period'].item()
        )

        # Update time related features (e.g., days to christmas)
        self.update_time_features(
            self._internal_data, 
            self.observation, 
            self.observation_params, 
            current_period=self.observation['current_period'].item() + 1
            )

        # print(f'current_demands: {current_demands[0, 0]}')
        
        # 2. Calculate post-demand inventory
        inventory = observation['store_inventories']
        inventory_on_hand = inventory[:, :, 0]
        post_inventory_on_hand = inventory_on_hand - current_demands

        # print(f'post_inventory_on_hand: {post_inventory_on_hand[0, 0]}')
        
        # 3. Calculate variable costs following base simulator pattern
        if self.maximize_profit:
            base_costs = (
                -observation['underage_costs'] * torch.minimum(inventory_on_hand, current_demands) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
            )
            # print(f'base_costs: {base_costs[0, 0]}')
        else:
            base_costs = (
                observation['underage_costs'] * torch.clip(-post_inventory_on_hand, min=0) + 
                observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
            )
        
        # Add procurement costs
        base_costs += observation['procurement_costs']*action_dict['feature_actions']['total_action']
        # print total action and demand and current period
        # print(f'total_action: {action_dict["feature_actions"]["total_action"][0, 0]}')
        # print(f'demand: {current_demands[0, 0]}')
        # print(f'current_period: {observation["current_period"]}')
        # Sum costs across stores
        base_costs = base_costs.sum(dim=1)
        
        # 4. Handle lost demand if needed
        if self.problem_params.get('lost_demand', False):
            post_inventory_on_hand = torch.clip(post_inventory_on_hand, min=0)
        
        # print(f'post_inventory_on_hand: {post_inventory_on_hand[0, 0]}')
        
        # 5. Get allocation from feature actions
        feature_actions = action_dict['feature_actions']
        allocation = feature_actions['total_action']
        # print(f'allocation: {allocation[0, 0]}')
        
        # 6. Update inventories using same pattern as base simulator
        lead_time = int(observation['lead_times'][0, 0].item())
        # raise an error if observation['lead_times'] contains more than one differentvalue
        if observation['lead_times'].unique().numel() > 1:
            raise ValueError('observation["lead_times"] contains more than one different value')
        # lead_time = self.store_params['lead_time']['value']
        # print(f'inventory: {observation["store_inventories"][0]}')
        # print(f'post_inventory_on_hand: {post_inventory_on_hand[0]}')
        # print(f'allocation: {allocation[0]}')
        # print(f'lead_time: {lead_time}')
        observation['store_inventories'] = self._update_inventories_in_place(
            inventory,
            post_inventory_on_hand,
            allocation,
            lead_time
        )
        # print(f'observation["store_inventories"]: {observation["store_inventories"][0]}')
        # print()

        # print(f'observation["store_inventories"]: {observation["store_inventories"][0, 0]}')
        # Update current period
        self.observation['current_period'] += 1

        return observation, base_costs
    
    def _calculate_additional_costs(self, observation, action_dict):
        """Calculate additional costs from registered components"""
        additional_costs = torch.zeros(self.batch_size, device=self.device)

        # Apply all registered cost functions
        for cost_func in self.cost_functions.values():
            additional_costs += cost_func(observation, action_dict)
        # print(f'additional_costs: {additional_costs[0]}')
        
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

    def _compute_normalization_stats(self):
        """Compute mean and std using collected inventory observations"""
        # Stack all observations along first dimension
        all_inventories = torch.stack(self._inventory_observations, dim=0)  # shape: [n_steps, batch_size, n_stores]
        
        # Compute stats across all dimensions
        self.inventory_mean = all_inventories.mean().item()
        self.inventory_std = all_inventories.std().item()
        if self.inventory_std == 0:
            self.inventory_std = 1.0

    def _normalize_observation(self, observation):
        """Normalize inventory levels in observation"""
        if not self.normalize_observations:
            return observation
            
        normalized_observation = observation.copy()
        store_inventory = observation['store_inventories']
        
        # Normalize inventory on hand
        normalized_inventory = (store_inventory - self.inventory_mean) / self.inventory_std
        
        # Update only the inventory on hand part
        normalized_observation['store_inventories'] = normalized_inventory
        
        return normalized_observation