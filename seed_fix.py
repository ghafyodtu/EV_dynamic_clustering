

def set_seeds(seed=42):
    from my_imports import rnd_, np
    # Set the seed for Python's built-in random module
    rnd_.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
