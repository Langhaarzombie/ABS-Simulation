import numpy as np
import scipy.constants as const
from scipy.stats import maxwell
from src.sphere import Sphere

def initialize(window_size, count, mass, temperature):
    """
    Initialize spheres for simulation.

    Create an array of spheres to be used in the simulation.

    Parameters
    ----------
    window_size: int
        Size of observed window, restricts location of spheres.
    count: int
        Number of spheres to be created.
    mass: int
        Mass of spheres used in velocity distribution.
    temperature:
        Temperature of the simulated system.

    Returns
    -------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    spheres = np.array([])
    for _ in np.arange(count):
        s = Sphere(window_size)

        # Distribute sphere in space
        loc_gen = np.random.rand(3)
        s.location = loc_gen * window_size

        # Assign velocity with Maxwell Boltzmann distribution
        scale = np.sqrt(const.k * temperature / mass)
        vel_gen = maxwell.rvs(size=3, scale=scale)
        s.velocity = vel_gen

        spheres = np.append(spheres, s)
    return spheres

def step(spheres, boundary, dt, file):
    """
    Calculate and save simulation step.

    Calculate the next step of the simulation of the given spheres.
    Append the result to the given file.

    Parameters
    ----------
    spheres: numpy.array of Sphere
        Array of ABS for simulation
    boundary: int
        Size of observed window.
    dt: int
        Size of timestep.
    file: _io.TextIOWrapper
        Already openend save file.

    Returns
    -------
    numpy.array of Sphere
        Updated array of ABS for simulation.
    """
    for s in spheres:
        s.location = s.location + dt * s.velocity
        print(f"{s.location[0]}\t{s.location[1]}\t{s.location[2]}", end="\n", file=file)
    return spheres

