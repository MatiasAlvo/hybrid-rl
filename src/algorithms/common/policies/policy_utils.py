import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_proportional_allocation(store_outputs, warehouse_inventories):
    """
    Apply proportional allocation feasibility enforcement function.
    Assigns inventory proportionally to store order quantities when warehouse inventory is insufficient.
    """
    total_limiting_inventory = warehouse_inventories[:, :, 0].sum(dim=1)
    sum_allocation = store_outputs.sum(dim=1)
    
    final_allocation = torch.multiply(
        store_outputs,
        torch.clip(total_limiting_inventory / (sum_allocation + 1e-12), max=1)[:, None]
    )
    return final_allocation

def apply_softmax_feasibility(store_outputs, warehouse_inventory, transshipment=False):
    """Apply softmax across store outputs and multiply by warehouse inventory"""
    total_warehouse_inv = warehouse_inventory[:, :, 0].sum(dim=1)
    softmax_inputs = store_outputs
    
    if not transshipment:
        softmax_inputs = torch.cat((
            softmax_inputs,
            torch.ones_like(softmax_inputs[:, 0])[:, None]
        ), dim=1)
    
    softmax_outputs = F.softmax(softmax_inputs, dim=1)
    
    if not transshipment:
        softmax_outputs = softmax_outputs[:, :-1]
    
    return torch.multiply(softmax_outputs, total_warehouse_inv[:, None])

def concatenate_with_context(object_state, context):
    """Concatenate context vector to every location's local state"""
    n_objects = object_state.size(1)
    context = context.unsqueeze(1).expand(-1, n_objects, -1)
    return torch.cat((object_state, context), dim=2)

# ... other utility functions from MyNeuralNetwork ... 