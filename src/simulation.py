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
    dt: float
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
        # Estimate old location
        s.old_location = s.location - s.velocity*dt

    return spheres

def step(spheres, boundary, sigma, epsilon, dt, file):
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
    sigma: float64
        Sigma in the potential.
    epsilon: float64
        Epsilon in the potential.
    dt: float
        Size of timestep.
    file: _io.TextIOWrapper
        Already openend save file.

    Returns
    -------
    numpy.array of Sphere
        Updated array of ABS for simulation.
    """
    diameter = boundary*np.sqrt(2)
    cut_off = (2**(1/6)*sigma)**2 # Cutoff distance for potential
    for i in np.arange(len(spheres)):
        s1 = spheres[i]
        # Loop over unique pairs
        for j in np.arange(i+1, len(spheres)):
            s2 = spheres[j]
            dist = s1.location - s2.location
            dist = dist - np.rint(dist/diameter) # periodic bonudaries
            r = np.inner(dist, dist)
            if r < cut_off:
                # Calculate acting force
                force = 4*epsilon * ((sigma**12/r**6) - (sigma**6/r**3)) + epsilon
                direction = dist / np.linalg.norm(dist)
                s1.force += force * direction
                s2.force -= force * direction
    # Verlet for updating locations
    for s in spheres:
        s.update(2*s.location - s.old_location + s.force * dt**2, dt)
        s.force = np.zeros(3)
        print(f"{s}", end="\n", file=file)
    return spheres

