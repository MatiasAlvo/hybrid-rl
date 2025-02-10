import scipy.stats as stats
import numpy as np

def long_run_avg_cost(s, S, K, h, p, lambda_):
    """
    Compute the long-run average cost for an (s, S) inventory policy in a periodic review system with 
    Poisson demand and zero lead time.
    
    Parameters:
    s       : Reorder point (continuous or integer)
    S       : Order-up-to level (continuous or integer)
    K       : Fixed ordering cost per order
    h       : Holding cost per unit per period
    p       : Underage (stockout) cost per unit short
    lambda_ : Mean demand per period (Poisson)
    
    Returns:
    Long-run average cost per period
    """
    
    # 1. Compute stationary probabilities of inventory levels
    P_D = stats.poisson.pmf(np.arange(S - s + 1), lambda_)  # Probabilities of demand realizations
    P_D /= P_D.sum()  # Normalize to get steady-state probabilities
    
    # 2. Expected order frequency (orders per period)
    lambda_o = sum(P_D)  # Since we order exactly once per cycle
    
    # 3. Expected holding cost
    expected_inventory = sum((s + i) * P_D[i] for i in range(len(P_D)))
    holding_cost = h * expected_inventory
    
    # 4. Expected stockout cost
    expected_shortage = sum(max(d - S, 0) * stats.poisson.pmf(d, lambda_) for d in range(int(3 * lambda_)))
    shortage_cost = p * expected_shortage
    
    # 5. Expected cycle length
    expected_cycle_length = sum(P_D)  # Since a cycle is completed when inventory reaches s
    
    # 6. Long-run average cost
    long_run_cost = (K * lambda_o / expected_cycle_length) + holding_cost + shortage_cost
    
    return long_run_cost

# Example usage
s, S, K, h, p, lambda_ = 6, 40, 24, 1, 9, 10
cost = long_run_avg_cost(s, S, K, h, p, lambda_)
print(f"Long-run average cost: {cost:.2f}")