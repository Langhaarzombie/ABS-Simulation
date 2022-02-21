import numpy as np
import scipy.constants as const
from scipy.stats import maxwell
from src.sphere import Sphere

def initialize(window_size, count, temperature, dt):
    """
    Initialize spheres for simulation.

    Create an array of spheres to be used in the simulation.
    Spheres are put in the window without overlap and are assigned
    a random velocity so that the total velocity is zero. The
    given temperature defines the velocity scaling.

    Parameters
    ----------
    window_size: int
        Size of observed window, restricts location of spheres.
    count: int
        Number of spheres to be created.
    temperature:
        Temperature of the simulated system.
    dt: int
        Size of timestep in simulation.

    Returns
    -------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    # Create lattice
    dl = np.ceil(window_size / np.cbrt(count))
    li = np.arange(window_size, step=dl)
    x, y, z = np.meshgrid(li, li, li)
    x = np.reshape(x, [-1, 1])
    y = np.reshape(y, [-1, 1])
    z = np.reshape(z, [-1, 1])
    lattice = np.concatenate((x, y, z), axis=1)

    # Generate spheres
    spheres = np.array([])
    mean_vel = np.zeros(3)
    mean_vel2 = 0
    for i in np.arange(count):
        s = Sphere(window_size)
        # Put sphere on lattice
        s.location = lattice[i]
        # Assign random velocity
        s.velocity = np.random.rand(3) - [0.5, 0.5, 0.5]

        mean_vel += s.velocity/count     # mean total velocity
        mean_vel2 += np.inner(s.velocity, s.velocity)/count # mean velocity squared

        spheres = np.append(spheres, s)

    temp_fac = np.sqrt(3*temperature/mean_vel2)
    for s in spheres:
        # Correct velocity for total vel = 0 and desired temperature
        s.velocity = (s.velocity - mean_vel) * temp_fac
        # Move sphere to previous timestep position
        s.location = s.location - s.velocity*dt

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

