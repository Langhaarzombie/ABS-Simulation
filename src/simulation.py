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

def step(spheres, boundary, sigma, temperature, dt):
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
    temperature: float32
        Temperature of the heat bath.
    dt: float32
        Size of timestep.

    Returns:
    --------
    numpy.array of Sphere
        Updated array of ABS for simulation.
    """
    # Velocity Verlet with Langevin thermostat for updating positions
    gamma = 100
    sig = 2

    etas = np.array([])
    xis = np.array([])
    for s in spheres:
        eta = np.random.normal(loc=0, scale=np.sqrt(2*temperature*gamma), size=3)
        xi = np.random.normal(loc=0, scale=np.sqrt(2*temperature*gamma), size=3)
        # v(t + dt/2)
        s.velocity = _v_half_step(s.velocity, dt, s.acceleration, sig, gamma, eta, xi)
        # x(t + dt)
        s.position += s.velocity * dt

        etas = np.append(etas, eta)
        xis = np.append(xis, xi)

    # forces
    locs = _get_positions(spheres)
    fs, pot_ens = calculate_forces(locs, boundary, sigma, dt)

    # v(t + dt)
    for i in np.arange(len(spheres)):
        s = spheres[i]
        s.acceleration = fs[i]
        s.potential_energy = pot_ens[i]
        s.velocity = _v_half_step(s.velocity, dt, s.acceleration, sig, gamma, etas[i], xis[i])

    return spheres

def _v_half_step(v, dt, f, sig, gamma, eta, xi):
    res = v + 0.5*dt*f
    res -= 0.5*dt*gamma*v
    res += 0.5*np.sqrt(dt)*sig*xi
    res -= 0.125*dt**2*gamma*(f-gamma*v)
    res -= 0.25*dt**(3/2)*gamma*sig*(0.5*xi+eta/np.sqrt(3))
    return res

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

