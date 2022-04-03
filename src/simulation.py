import numpy as np
import scipy.constants as const
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
    dt: float32
        Size of timestep in simulation.

    Returns
    -------
    numpy.array of Sphere
        Array of ABS for simulation.
    """
    # Create lattice
    dl = np.round(window_size / np.cbrt(count))
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

def calculate_forces(locations, boundary, sigma, dt):
    """
    Calculate forces and potential energy for spheres at locations.

    Calculate forces acting on sphere at location and potential energy.

    Parameters:
    -----------
    locations: numpy.array of float64
        Locations of spheres.
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
    forces = np.zeros_like(locations)
    pot_ens = np.zeros(len(locations))

    diam = boundary*np.sqrt(2)
    cut_off = (2**(1/6)*sigma)**2 # Cutoff distance for potential
    s6 = sigma**6
    ecut = 4*s6*(s6**2/cut_off**6 - 1/cut_off**3)

    for i in np.arange(len(locations)):
        # Loop over unique pairs
        for j in np.arange(i+1, len(locations)):
            dist = locations[i] - locations[j]
            dist = dist - boundary * np.rint(dist/boundary) # periodic bonudaries
            r2 = np.inner(dist, dist)
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
        s.location += s.velocity * dt + 0.5 * s.acceleration * dt**2

    # a(t + dt) and v(t + dt)
    locs = _get_locations(spheres)
    fs, pot_ens = calculate_forces(locs, boundary, sigma, dt)

    for i in np.arange(len(spheres)):
        s = spheres[i]
        s.velocity += 0.5 * dt * (s.acceleration + fs[i])
        s.acceleration = fs[i]
        s.potential_energy = pot_ens[i]
        print(f"{s}", end="\n", file=file)

    return spheres

def _get_locations(spheres):
    """
    Get list of locations from list of spheres.

    Extracts location information from spheres and gives list of pure floats.

    Parameters
    ----------
    spheres: numpy.array of Sphere
        Array of ABS for simulation.
    Returns:
    --------
    numpy.array of float64
        Array of locations of ABSs.
    """
    loc = np.empty((0, 3), float)
    for s in spheres:
        loc = np.append(loc, [s.location], axis=0)
    return loc

