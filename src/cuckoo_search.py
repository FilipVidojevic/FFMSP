import numpy as np

EPS = 10**-5

def Michaelwicz_function(x, y, m=10):
    """Michaelwicz function (min at (2.20, 1.57) for m=10)"""
    return -np.sin(x) * (np.sin(x**2 / np.pi))**(2 * m) - np.sin(y) * (np.sin(2 * y**2 / np.pi))**(2 * m)

def De_Jong(x, y):
    """De Jong's function (min at x=0)"""
    return x**2 + y**2


def powerlaw_step(beta=1.5, l_min=1.0):
    """
    Generate a step length from a power-law distribution with heavy tail.
    
    P(l) ∝ l^-(1 + beta), for l >= l_min
    
    Args:
        beta (float): power-law exponent (0 < beta < 2)
        l_min (float): minimum step length

    Returns:
        float: a step length sampled from the distribution
    """
    u = np.random.uniform(0, 1)
    return l_min * (u ** (-1 / beta))

def cuckoo_search(obj_func, n_nests=10, n_iter=10000, alpha=1, pa=0.1, bounds=[[-5.12, 5.12], [-5.12, 5.12]]):
    """Basic Cuckoo Search algorithm for optimizing f(x, y)"""
    # Initialize nests randomly
    nests = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_nests, 2))
    fitness = np.array([obj_func(x, y) for x, y in nests])

    for _ in range(n_iter):
        for i in range(n_nests):
            # Generate new solution via Lévy flight
            step = powerlaw_step()
            new_nest = nests[i] + alpha * step
            # Clip to bounds
            new_nest = np.clip(new_nest, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fitness = obj_func(new_nest[0], new_nest[1])
            
            # Greedy selection
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness
        
        # Abandon a fraction of worst nests and replace with new random ones
        n_abandon = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[-n_abandon:]
        nests[worst_indices] = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(n_abandon, 2))
        for i in worst_indices:
            fitness[i] = obj_func(nests[i][0], nests[i][1])

    # Return the best solution found
    best_index = np.argmin(fitness)
    return nests[best_index], fitness[best_index]

