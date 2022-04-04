import numpy as np
import scipy.constants as const
from numba import njit
from src.sphere import Sphere

def initialize(bounds, count, temperature, dt):
    """
    Initialize spheres for simulation.

    Create an array of spheres to be used in the simulation.
    Spheres are put in the window without overlap and are assigned
    a random velocity so that the total velocity is zero. The
    given temperature defines the velocity scaling.

    Parameters:
    -----------
    bounds: int
        Length of observed cubic window.
    count: int
        Number of spheres to be created.
    temperature:
        Temperature of the simulated system.
    dt: float32
        Size of timestep in simulation.

    Returns:
    --------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    # Create lattice and generate spheres
    spheres, mean_vel, mean_vel2 = generate_spheres(bounds, count)

    # Readjust velocities acc. to temp and total_momentum = 0
    temp_fac = np.sqrt(3*temperature/mean_vel2)
    for s in spheres:
        s.velocity = (s.velocity - mean_vel) * temp_fac

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
                    s = Sphere(bounds)
                    s.position = origin + c
                    s.velocity = np.random.rand(3) - [0.5, 0.5, 0.5]
                    mean_vel += s.velocity/count     # mean total velocity
                    mean_vel2 += np.dot(s.velocity, s.velocity)/count # mean velocity squared
                    spheres = np.append(spheres, s)
                    if len(spheres) == count:
                        return spheres, mean_vel, mean_vel2
    return spheres, mean_vel, mean_vel2

@njit
def calculate_forces(positions, boundary, sigma, dt):
    """
    Calculate forces and potential energy for spheres at positions.

    Calculate forces acting on sphere at positions and potential energy.

    Parameters:
    -----------
    positions: numpy.array of float64
        Positions of spheres.
    boundary: int
        Size of simulation window.
    sigma: float32
        Sigma for potential.
    dt: float32
        Size of timestep in simulation.

    Returns:
    --------
    forces: 2d numpy.array of float64
        Forces acting on spheres.
    pot_ens: numpy.array of float64
        Potential energies of spheres.
    """
    forces = np.zeros_like(positions)
    pot_ens = np.zeros(len(positions))

    diam = boundary*np.sqrt(2)
    cut_off = (2**(1/6)*sigma)**2 # Cutoff distance for potential
    s6 = sigma**6
    ecut = 4*s6*(s6**2/cut_off**6 - 1/cut_off**3)

    for i in np.arange(len(positions)):
        # Loop over unique pairs
        for j in np.arange(i+1, len(positions)):
            dist = positions[i] - positions[j]
            dist = dist - boundary * np.rint(dist/boundary) # periodic bonudaries
            r2 = np.dot(dist, dist)
            if r2 < cut_off:
                # Calculate acting force, Lennard Jones potential
                r2d = 1/r2
                r6d = r2d**3
                force = 48*r2d*r6d*s6*(r6d*s6-0.5)
                forces[i] += force * dist
                forces[j] -= force * dist
                pot_ens[i] += 2*r6d*s6*(r6d*s6-1) - ecut
                pot_ens[j] += 2*r6d*s6*(r6d*s6-1) - ecut
    return forces, pot_ens

def step(spheres, boundary, sigma, dt, file):
    """
    Calculate and save simulation step.

    Calculate the next step of the simulation of the given spheres.
    Append the result to the given file.

    Parameters
    ----------
    spheres: numpy.array of Sphere
        Array of ABS for simulation.
    boundary: int
        Size of observed window.
    sigma: float32
        Sigma in the potential.
    dt: float32
        Size of timestep.
    file: _io.TextIOWrapper
        Already openend save file.

    Returns
    -------
    numpy.array of Sphere
        Updated array of ABS for simulation.
    """
    # Velocity Verlet for updating positions
    # x(t + dt)
    for s in spheres:
        s.position += s.velocity * dt + 0.5 * s.acceleration * dt**2

    # a(t + dt) and v(t + dt)
    locs = _get_positions(spheres)
    fs, pot_ens = calculate_forces(locs, boundary, sigma, dt)

    for i in np.arange(len(spheres)):
        s = spheres[i]
        s.velocity += 0.5 * dt * (s.acceleration + fs[i])
        s.acceleration = fs[i]
        s.potential_energy = pot_ens[i]
        print(f"{s}", end="\n", file=file)

    return spheres

def _get_positions(spheres):
    """
    Get list of positions from list of spheres.

    Extracts positions information from spheres and gives list of pure floats.

    Parameters
    ----------
    spheres: numpy.array of Sphere
        Array of ABS for simulation.
    Returns:
    --------
    numpy.array of float64
        Array of positions of ABSs.
    """
    loc = np.zeros((0, 3), float)
    for s in spheres:
        loc = np.append(loc, [s.position], axis=0)
    return loc

