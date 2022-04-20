import numpy as np
from numba import njit
from src.sphere import Sphere

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
        forces_i = forces[i]
        pot_ens_i = pot_ens[i]
        for j in np.arange(i+1, len(positions)):
            dist = positions[i] - positions[j]
            dist = dist - boundary * np.rint(dist/boundary) # periodic bonudaries
            r2 = np.dot(dist, dist)
            if r2 < cut_off:
                # Calculate acting force, Lennard Jones potential
                r2d = 1/r2
                r6d = r2d**3
                force = 48*r2d*r6d*s6*(r6d*s6-0.5)
                forces_i += force * dist
                pot_ens_i += 2*r6d*s6*(r6d*s6-1) - ecut
                forces[j] -= force * dist
                pot_ens[j] += 2*r6d*s6*(r6d*s6-1) - ecut
        forces[i] = forces_i
        pot_ens[i] = pot_ens_i
    return forces, pot_ens

def step(spheres, boundary, sigma, dt):
    """
    Calculate and save simulation step.

    Calculate the next step of the simulation of the given spheres.
    Append the result to the given file.

    Parameters:
    -----------
    spheres: numpy.array of Sphere
        Array of ABS for simulation.
    boundary: int
        Size of observed window.
    sigma: float32
        Sigma in the potential.
    dt: float32
        Size of timestep.

    Returns:
    --------
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

    return spheres

def _get_positions(spheres):
    """
    Get list of positions from list of spheres.

    Extracts positions information from spheres and gives list of pure floats.

    Parameters:
    -----------
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

