import numpy as np
from numba import njit
from src.sphere import Sphere
from src.writer import Writer

def random(config):
    """
    Initialize spheres for simulation with random distribution.

    Create an array of spheres to be used in the simulation.
    Spheres are put in the window without overlap and are assigned
    a random velocity so that the total velocity is zero. The
    given temperature defines the velocity scaling.

    Parameters:
    -----------
    config: dict
        Config read from yaml config file.

    Returns:
    --------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    # Create lattice and generate spheres
    bounds = np.cbrt(config["count"]/config["density"])
    spheres, mean_vel, mean_vel2 = generate_spheres(bounds, config["count"])

    # Readjust velocities acc. to temp and total_momentum = 0
    temp_fac = np.sqrt(3*config["temperature"]/mean_vel2)
    for s in spheres:
        s.velocity = (s.velocity - mean_vel) * temp_fac

    # Save init to file
    saviour = Writer(config, ["position", "velocity", "active_acceleration", "bounds"])
    saviour.write(spheres)
    saviour.close_file()

    return spheres, bounds

def from_file(file):
    """
    Initialize spheres for simulation from saved configuration.

    Create an array of spheres to be used in the simulation.
    All parameters (position, velocity, bounds, count and
    timestep) are read from the given file.

    Parameters:
    -----------
    file: str
        Filename of init file.

    Returns:
    --------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    data = np.genfromtxt(file, names=True)
    spheres = np.array([])
    for d in data:
        bounds = d["b"]
        pos = np.array([d["x"], d["y"], d["z"]])
        vel = np.array([d["vx"], d["vy"], d["vz"]])
        act = np.array([d["ax"], d["ay"], d["az"]])
        s = Sphere(bounds, pos, vel, act)
        spheres = np.append(spheres, s)
    return spheres


def generate_spheres(bounds, count):
    """
    Put spheres on fcc lattice, assign random velocity.

    Generates fcc lattice and assigns locations to spheres.
    Assigns randomly distributed velocities to spheres (total momentum != 0).

    Parameters:
    -----------
    bounds: int
        Length of observed cubic window.
    count: int
        Number of spheres to generate.

    Returns:
    --------
    numpy.array of Sphere
        Array of ABS with assigned locations.
    float64
        Mean velocity of all spheres.
    float64
        Mean velocity squared of all spheres.
    """
    cpd = np.ceil(np.cbrt(count / 4)) # cells per direction x, y, z
    a = bounds / cpd # lattice constant
    basis = a * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cell = a * np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])

    spheres = np.array([])
    mean_vel = np.zeros(3)
    mean_vel2 = 0

    indices = np.arange(cpd)
    for i in indices:
        for j in indices:
            for k in indices:
                origin = np.dot(basis.T, [i, j, k])
                for c in cell:
                    pos = origin + c
                    vel = np.random.rand(3) - [0.5, 0.5, 0.5]
                    act = np.random.rand(3) - [0.5, 0.5, 0.5]
                    s = Sphere(bounds, pos, vel, act / np.linalg.norm(act))
                    mean_vel += s.velocity/count     # mean total velocity
                    mean_vel2 += np.dot(s.velocity, s.velocity)/count # mean velocity squared
                    spheres = np.append(spheres, s)
                    if len(spheres) == count:
                        return spheres, mean_vel, mean_vel2
    return spheres, mean_vel, mean_vel2

