import math
import numpy as np

def simulate_inventory_backup(p, h, lambd, K, s, S, steps=1000000):
    inventory = S  # Start at the upper threshold
    total_cost = 0
    
    for _ in range(steps):
        demand = np.random.poisson(lambd)
        inventory -= demand
        
        # Compute holding and shortage costs
        holding_cost = h * max(inventory, 0)
        shortage_cost = p * max(-inventory, 0)
        
        # If inventory falls below s, order up to S
        if inventory <= s:
            total_cost += K  # Fixed ordering cost
            inventory = S
        
        total_cost += holding_cost + shortage_cost
    
    return total_cost / steps

def simulate_inventory(p, h, lambd, K, s, S, L, steps=1000000):
    inventory = S  # Start at the upper threshold
    pipeline = [0] * (L + 1)  # Track arriving orders
    total_cost = 0
    
    for _ in range(steps):
        # print(f'Inventory: {inventory}')
        # print(f'Pipeline: {pipeline}')
        demand = np.random.poisson(lambd)
        # print(f'Demand: {demand}')
        inventory -= demand
        
        # Receive incoming order from L periods ago
        inventory += pipeline.pop(0)
        pipeline.append(0)
        
        # Compute holding and shortage costs
        holding_cost = h * max(inventory, 0)
        shortage_cost = p * max(-inventory, 0)
        
        # If inventory falls below s, place order up to S
        sum_inventory = sum(pipeline) + inventory
        if sum_inventory <= s:
            total_cost += K  # Fixed ordering cost
            order_quantity = S - sum_inventory
            # print(f'Ordering {order_quantity}')
            pipeline[-1] = order_quantity  # Schedule arrival in L periods
        # print()
        
        
        total_cost += holding_cost + shortage_cost
    
    return total_cost / steps


# def compute_m_M(lambd, n):
#     m = [0] * (n + 1)
#     M = [0] * (n + 1)
#     m[0] = 1 / (1 - math.exp(-lambd))
#     for j in range(1, n + 1):
#         m[j] = sum((lambd ** i * math.exp(-lambd) / math.factorial(i)) * m[j - i] for i in range(1, j + 1))
#         M[j] = M[j - 1] + m[j - 1]
#     return m, M

# def G(y, h, p, lambd, L):
#     effective_lambd = lambd * (L + 1)
#     term1 = sum((y - k) * (effective_lambd ** k * math.exp(-effective_lambd) / math.factorial(k)) for k in range(y + 1))
#     term2 = sum((k - y) * (effective_lambd ** k * math.exp(-effective_lambd) / math.factorial(k)) for k in range(y + 1, 100))  # Truncate at 100
#     return h * term1 + p * term2

# def c(s, S, h, p, lambd, K, L):
#     m, M = compute_m_M(lambd, S - s)
#     numerator = K + sum(m[j] * G(S - j, h, p, lambd, L) for j in range(S - s))
#     denominator = M[S - s]
#     return numerator / denominator

# def find_optimal_sS(h, p, lambd, K, L):
#     y_star = 0
#     while G(y_star + 1, h, p, lambd) < G(y_star, h, p, lambd):
#         y_star += 1
    
#     S0 = y_star
#     s = y_star
#     while c(s - 1, S0, h, p, lambd, K, L) < G(s - 1, h, p, lambd*(L+1)):
#         s -= 1
#     s0 = s
#     co = c(s0, S0, h, p, lambd, K, L)
#     S = S0 + 1
    
#     while G(S, h, p, lambd*(L+1)) < co:
#         if c(s, S, h, p, lambd, K, L) < co:
#             S0 = S
#             while c(s, S0, h, p, lambd, K, L) > G(s + 1, h, p, lambd*(L+1)):
#                 s += 1
#             co = c(s, S0, h, p, lambd, K, L)
#         S += 1
    
#     return s, S0, co

def compute_m_M(lambd, n):
    if n < 0:
        return [1], [1]  # Handle invalid cases gracefully
    m = [0] * (n + 1)
    M = [0] * (n + 1)
    m[0] = 1 / (1 - math.exp(-lambd)) if lambd != 0 else 1  # Avoid division by zero
    for j in range(1, n + 1):
        m[j] = sum((lambd ** i * math.exp(-lambd) / math.factorial(i)) * m[j - i] for i in range(1, j + 1))
        M[j] = M[j - 1] + m[j - 1]
    return m, M

def G(y, h, p, lambd, L):
    print(f'y: {y}')
    print(f'lambd: {lambd}')
    print(f'L: {L}')
    effective_lambd = lambd * (L + 1)
    term1 = sum((y - k) * (effective_lambd ** k * math.exp(-effective_lambd) / math.factorial(k)) for k in range(y + 1))
    term2 = sum((k - y) * (effective_lambd ** k * math.exp(-effective_lambd) / math.factorial(k)) for k in range(y + 1, 100))  # Truncate at 100
    return h * term1 + p * term2

def c(s, S, h, p, lambd, K, L):
    if S <= s:
        return float('inf')  # Avoid invalid computation
    m, M = compute_m_M(lambd, S - s)
    denominator = M[S - s] if M[S - s] != 0 else 1  # Avoid division by zero
    numerator = K + sum(m[j] * G(S - j, h, p, lambd, L) for j in range(S - s))
    return numerator / denominator

def find_optimal_policy(h, p, lambd, K, L):
    """
    Find optimal (s*, S*) policy using algorithm from Section 3
    
    Parameters:
    h (float): Holding cost per unit per period
    p (float): Penalty cost per unit per period
    lambd (float): Mean demand rate (Poisson parameter)
    K (float): Fixed ordering cost
    L (int): Lead time
    
    Returns:
    tuple: (s*, S*, c*) - optimal reorder point, order-up-to level, and cost
    """
    
    def find_y_star():
        """Find minimum point of G function"""
        # Start search at mean lead time demand
        y = int(lambd * (L + 1))
        # Search downward
        while y > 0 and G(y-1, h, p, lambd, L) < G(y, h, p, lambd, L):
            y -= 1
        # Search upward if necessary
        while G(y+1, h, p, lambd, L) < G(y, h, p, lambd, L):
            y += 1
        return y
    
    # Step 0: Initialize
    y_star = find_y_star()
    s = y_star
    S_0 = y_star
    
    # Find initial s_0 by decreasing s until c(s,S_0) ≤ G(s)
    while True:
        if s <= 0:
            break
        cost = c(s-1, S_0, h, p, lambd, K, L)
        if cost >= G(s-1, h, p, lambd, L):
            break
        s -= 1
    
    s_0 = s
    c_0 = c(s_0, S_0, h, p, lambd, K, L)
    S = S_0 + 1
    
    # Step 1: Main optimization loop
    while G(S, h, p, lambd, L) < c_0:
        current_cost = c(s, S, h, p, lambd, K, L)
        
        if current_cost < c_0:
            S_0 = S
            # Update s while c(s,S_0) > G(s+1)
            while True:
                if c(s, S_0, h, p, lambd, K, L) <= G(s+1, h, p, lambd, L):
                    break
                s += 1
            c_0 = c(s, S_0, h, p, lambd, K, L)
        
        S += 1
    
    return s_0, S_0, c_0

def get_policy_evaluation(h, p, lambd, K, L):
    """
    Wrapper function that evaluates the policy and provides detailed output
    
    Returns dictionary with full policy details
    """
    s_star, S_star, c_star = find_optimal_policy(h, p, lambd, K, L)
    
    return {
        "s*": s_star,
        "S*": S_star,
        "Average Cost": c_star,
        "Order Quantity": S_star - s_star,
        "Lead Time Demand": lambd * (L + 1)
    }

# Example usage:
p = 9
h = 1
lambd = 10
K = 64
s = 6
S = 40
L = 0


print(simulate_inventory(p, h, lambd, K, s, S, L))
print(c(s, S, h, p, lambd, K, L))

policy = get_policy_evaluation(h, p, lambd, K, L)
print(f"Optimal policy: s* = {policy['s*']}, S* = {policy['S*']}")
print(f"Average Cost: {policy['Average Cost']:.2f}")

# s_opt, S_opt, cost_opt = find_optimal_sS(h, p, lambd, K, L)
# print(f"Optimal (s, S): ({s_opt}, {S_opt}), Cost: {cost_opt}")