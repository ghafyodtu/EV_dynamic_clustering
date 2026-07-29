def set_seeds():
    from my_imports import rnd_, np, tf
    # Set the seed for Python's built-in random module
    rnd_.seed(42)
    # Set the seed for NumPy
    np.random.seed(42)
    # Set the seed for TensorFlow
    tf.random.set_seed(42)