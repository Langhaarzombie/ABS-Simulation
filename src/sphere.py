import numpy as np

class Sphere:
    """
    Is an Active Brownian Sphere in the simulation.

    Attributes
    ----------
    location: numpy.ndarray of int
        x, y, z coordinates
    velocity: numpy.ndarray of int
        velocity in x, y, z direction
    """
    location = np.array([0, 0, 0])
    velocity = np.array([0, 0, 0])

