import math
from scipy.stats import poisson
import numpy as np

# def compute_m_M(lambd, n):
#     if n < 0:
#         return [1], [1]
#     m = [0] * (n + 1)
#     M = [0] * (n + 1)
#     P = [poisson.pmf(i, lambd) for i in range(n + 1)]
    
#     m[0] = 1 / (1 - P[0])  # Zheng's formula for m(0)
#     for j in range(1, n + 1):
#         m[j] = sum(P[i] * m[j - i] for i in range(1, j + 1))  # Recursive formula
#         M[j] = M[j - 1] + m[j - 1]
#     return m, M

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

from scipy.stats import poisson
import numpy as np

def G(y, h, p, lambd, L):
    if y < 0:
        return 0  # Avoid invalid calculations
    
    effective_lambd = float(lambd) * (L + 1)
    
    # More conservative truncation: cover 99.999% probability mass
    truncation_limit = int(poisson.ppf(0.99999, effective_lambd)) + 1
    k_range = np.arange(truncation_limit, dtype=np.float64)
    
    # Compute Poisson PMF with high precision
    pmf = poisson.pmf(k_range, effective_lambd).astype(np.float64)
    
    # Ensure valid indexing
    max_k = min(y + 1, len(pmf))
    
    # Compute cost terms
    term1 = np.sum((y - k_range[:max_k]) * pmf[:max_k])
    term2 = np.sum((k_range[max_k:] - y) * pmf[max_k:]) if max_k < len(pmf) else 0
    
    # Debugging: Print intermediate values for analysis
    # print(f"y={y}, term1={term1}, term2={term2}, h*term1={h*term1}, p*term2={p*term2}")

    return h * term1 + p * term2


def G_backup(y, h, p, lambd, L):
    effective_lambd = lambd * (L + 1)
    # Use scipy.stats.poisson for numerical stability
    pmf = poisson.pmf(np.arange(int(effective_lambd * 10)), effective_lambd)
    
    # Calculate terms using vectorized operations
    k_range = np.arange(int(effective_lambd * 10))
    term1 = np.sum((y - k_range[:y+1]) * pmf[:y+1])
    term2 = np.sum((k_range[y+1:] - y) * pmf[y+1:])
    
    return h * term1 + p * term2

def c(s, S, h, p, lambd, K, L):
    if S <= s:
        return float('inf')  # Avoid invalid computation
    m, M = compute_m_M(lambd, S - s)
    denominator = M[S - s] if M[S - s] != 0 else 1  # Avoid division by zero
    numerator = K + sum(m[j] * G(S - j, h, p, lambd, L) for j in range(S - s))
    return numerator / denominator

def find_y_star(h, p, lambd, L):
    """Find minimum point of G function"""
    # Start from mean lead time demand
    y = int(lambd * (L + 1))
    # Search downward
    while y > 0 and G(y-1, h, p, lambd, L) < G(y, h, p, lambd, L):
        y -= 1
    # Search upward if necessary
    while G(y+1, h, p, lambd, L) < G(y, h, p, lambd, L):
        y += 1
    return y

def find_optimal_policy(h, p, lambd, K, L):
    """
    Find optimal (s*, S*) policy following the exact algorithm
    """
    # Step 0
    y_star = find_y_star(h, p, lambd, L)
    # print(f'y_star: {y_star}')
    s = y_star
    S_0 = y_star
    
    # Repeat s := s - 1 until c(s, S0) ≤ G(s)
    while c(s, S_0, h, p, lambd, K, L) > G(s, h, p, lambd, L):
        s = s - 1
        # print(f's: {s}')
        # print(f'c(s, S_0): {c(s, S_0, h, p, lambd, K, L)}')
        # print(f'G(s): {G(s, h, p, lambd, L)}')
        # print()
        
    s_0 = s
    c_0 = c(s_0, S_0, h, p, lambd, K, L)
    S = S_0 + 1
    
    # Step 1
    while G(S, h, p, lambd, L) <= c_0:
        if c(s, S, h, p, lambd, K, L) < c_0:
            S_0 = S
            # While c(s, S0) > G(s+1) do s := s+1
            while c(s, S_0, h, p, lambd, K, L) <= G(s+1, h, p, lambd, L):
                s = s + 1
            c_0 = c(s, S_0, h, p, lambd, K, L)
        S = S + 1
        
    return s, S_0, c_0

def simulate_inventory(p, h, lambd, K, s, S, L, steps=1000000):
    inventory = 0  # Start at the lower threshold
    pipeline = [0] * (L + 1)  # Track arriving orders
    total_cost = 0

    for _ in range(steps):
        demand = np.random.poisson(lambd)
        inventory -= demand  # Process demand
        
        # Receive incoming order from L periods ago
        inventory += pipeline.pop(0)
        pipeline.append(0)
        
        # Compute inventory position after arrivals
        sum_inventory = inventory + sum(pipeline)
        
        # Compute holding and shortage costs
        holding_cost = h * max(inventory, 0)
        shortage_cost = p * max(-inventory, 0)
        
        # If inventory position is at or below s, place order to S
        if sum_inventory <= s:
            total_cost += K  # Fixed ordering cost
            order_quantity = S - sum_inventory
            pipeline[-1] = order_quantity  # Schedule arrival in L periods
        
        total_cost += holding_cost + shortage_cost
    
    return total_cost / steps

def test_comparison(test_cases):
    """
    Compare simulation vs analytical results for various test cases
    """
    results = []
    for case in test_cases:
        h, p, lambd, K, s, S, L = case
        
        # Run simulation multiple times to get confidence interval
        sim_runs = 5
        sim_costs = []
        for _ in range(sim_runs):
            sim_cost = simulate_inventory(p, h, lambd, K, s, S, L, steps=100000)
            sim_costs.append(sim_cost)
            
        avg_sim_cost = np.mean(sim_costs)
        std_sim_cost = np.std(sim_costs)
        
        # Get analytical cost
        analytical_cost = c(s, S, h, p, lambd, K, L)
        
        # Calculate relative difference
        rel_diff = (avg_sim_cost - analytical_cost) / analytical_cost
        
        results.append({
            'params': case,
            'sim_cost': avg_sim_cost,
            'sim_std': std_sim_cost,
            'analytical_cost': analytical_cost,
            'rel_diff': rel_diff
        })
        
    return results


h = 1    # holding cost
p = 9    # penalty cost
# K = 30   # fixed ordering cost
# K = 30   # fixed ordering cost (64 is the value used in Veinott, also in Zheng though they report it incorrectly as 24)
lambd = 10  # demand rate
L = 2   # lead time
s = 26
S = 62

# K = 30
# Average Cost: 27.15
# K = 40
# Average Cost: 30.42
# K = 50
# Average Cost: 33.36
# K = 60
# Average Cost: 36.05
# K = 64
# Average Cost: 37.06
# K = 70
# Average Cost: 38.51
# K = 80
# Average Cost: 40.83

for K in [1, 30, 40, 50, 60, 64, 70, 80]:
    print(f"K = {K}")
    # print(c(s, S, h, p, lambd, K, L))

    s_star, S_star, c_star = find_optimal_policy(h, p, lambd, K, L)
    print(f"Optimal policy: s* = {s_star}, S* = {S_star}")
    print(f"Average Cost: {c_star:.2f}")
    cost_check = simulate_inventory(p, h, lambd, K, s, S, L)
    # if difference is too large, print the difference
    if abs(cost_check - c_star)/c_star > 0.1:
        print(f"Difference: {cost_check - c_star:.2f}")
    print()

test_cases = [
    # h, p, lambd, K, s, S, L
    (1, 9, 10, 64, 6, 40, 0),  # Base case from paper
    (1, 9, 10, 64, 6, 40, 1),  # With lead time
    (1, 9, 10, 64, 6, 40, 2),  # Longer lead time
    (1, 9, 5, 64, 6, 40, 0),   # Lower demand
    (1, 9, 20, 64, 6, 40, 0),  # Higher demand
    (1, 19, 10, 64, 6, 40, 0), # Higher penalty cost
    (2, 9, 10, 64, 6, 40, 0),  # Higher holding cost
]

# results = test_comparison(test_cases)

# # Print results in a formatted way
# for r in results:
#     params = r['params']
#     print(f"\nTest case: h={params[0]}, p={params[1]}, lambda={params[2]}, K={params[3]}, s={params[4]}, S={params[5]}, L={params[6]}")
#     print(f"Simulation cost: {r['sim_cost']:.2f} ± {r['sim_std']:.2f}")
#     print(f"Analytical cost: {r['analytical_cost']:.2f}")
#     print(f"Relative difference: {r['rel_diff']*100:.2f}%")