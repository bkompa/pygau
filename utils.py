import jax
import jax.numpy as np 

def squared_distance(x1, x2):
    return -2. * np.dot(x1, x2.T) + np.sum(x2**2, axis=1) + np.sum(x1**2, axis=1)[:, None]



