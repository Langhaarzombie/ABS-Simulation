import numpy as np
from numba import njit

def energy(config, data):
    """
    Calculate total, kinetic and potential energy during simulation for plot.

    Create the plot data to check energy conservation and stability of the system.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    Returns:
    --------
    ts: numpy.array of int
        Timesteps in simulation.
    potential: numpy.array of float64
        Total potential energy over time.
    kinetic: numpy.array of float64
        Total kinetic energy over time.
    total_energy: numpy.array of float64
        Total energy over time.
    """
    ts = np.arange(config["steps"])
    potential = np.array([])
    kinetic = np.array([])
    for i in ts:
        istart = config["count"] * i
        p = np.sum(data[istart:istart + config["count"]]["pen"])
        k = np.sum(data[istart:istart + config["count"]]["ken"])
        potential = np.append(potential, p)
        kinetic = np.append(kinetic, k)

    total_energy = kinetic + potential

    print(f"(std / mean) of total energy: {np.std(total_energy) / np.mean(total_energy)}")

    return ts, potential, kinetic, total_energy

def momentum(config, data):
    """
    Get momenta behaviour over time.

    Give the time and momenta of center of mass in x, y, z direction
    over time of simulation.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data: 2d numpy.ndarray of float64
        CSV data of simulation.
    Returns:
    --------
    ts: numpy.array of int
        Timesteps in simulation.
    momentum_x: numpy.array of float64
        Momentum of center of mass in x direction.
    momentum_y: numpy.array of float64
        Momentum of center of mass in y direction.
    momentum_z: numpy.array of float64
        Momentum of center of mass in z direction.
    """
    ts = np.arange(config["steps"])
    momentum_x = np.array([])
    momentum_y = np.array([])
    momentum_z = np.array([])
    for i in ts:
        istart = config["count"] * i
        momentum_x = np.append(momentum_x, np.sum(data[istart:istart + config["count"]]["vx"]))
        momentum_y = np.append(momentum_y, np.sum(data[istart:istart + config["count"]]["vy"]))
        momentum_z = np.append(momentum_z, np.sum(data[istart:istart + config["count"]]["vz"]))
    return ts, momentum_x, momentum_y, momentum_z

def temperature(config, data):
    """
    Get reduced temperature over time.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data: 2d numpy.ndarray of float64
        CSV data of simulation.
    Returns:
    --------
    ts: numpy.array of int
        Timesteps in simulation.
    temp: numpy.array of float64
        Temperature of simulated system.
    """
    ts = np.arange(config["steps"])
    temps = np.array([])
    for i in ts:
        istart = config["count"] * i
        temp = np.sum(data[istart:istart + config["count"]]["temp"])
        temps = np.append(temps, temp)
    return ts, temps

def radial_distribution(config, data, bin_count=1000):
    """
    Get radial distribution of spheres averaged over simulation time.

    Calculate the radial distribution g(r) of spheres in the box and
    average it over time where r is the distance.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data: 2d numpy.ndarray of float64
        CSV data of simulation.
    bin_count: int
        Resolution of r.
    Returns:
    --------
    r: numpy.array of float
        Distance from sphere.
    g: numpy.array of float
        Radial distribution values.
    """
    b = config["bounds"]
    g = np.zeros(bin_count)
    for i in np.arange(config["steps"]):
        istart = config["count"] * i
        spheres = data[istart:istart + config["count"]]
        g += _rad_distr(spheres, b, bin_count) / config["steps"]
    return np.linspace(0, b/2, bin_count, endpoint=True), g

@njit
def _rad_distr(spheres, bounds, binn):
    """
    Calculate radial distribution for one timestep.

    Parameters:
    -----------
    spheres: numpy.array of src.Sphere
        Spheres data in that timestep.
    bounds: int
        Bounds of box observed.
    binn: int
        Number of bins. Resolution of r.
    Returns:
    --------
    g: numpy.array of float
        Radial distribution values.
    """
    bcut = bounds / 2
    bins = bounds / (2*binn)
    g = np.zeros(binn)
    for i in np.arange(len(spheres)):
        xi = np.array([spheres[i]["x"], spheres[i]["y"], spheres[i]["z"]])
        for j in np.arange(i+1, len(spheres)):
            xj = np.array([spheres[j]["x"], spheres[j]["y"], spheres[j]["z"]])
            dist = xi - xj
            dist = dist - bounds * np.rint(dist/bounds)
            r = np.sqrt(np.dot(dist, dist))
            if r < bcut:
                ig = int(r/bins)
                g[ig] += 2
    for k in np.arange(binn):
        vol = ((k+1)**3 - k**3) * bins**3
        ppv = (len(spheres) / bounds**3) * vol * (4/3) * np.pi
        g[k] = g[k] / (ppv * len(spheres))
    return g

def velocity_correlation(config, data):
    """
    Calculate the average particle velocity correlation over time.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data: 2d numpy.ndarray of float64
        CSV data of simulation.
    Returns:
    --------
    ts: numpy.array of int
        Timesteps in simulation.
    correls: numpy.array of float64
        Average velocity correlations over time.
    """
    ts = np.arange(config["steps"])
    correls = np.array([])
    for i in ts:
        istart = config["count"] * i
        correl = np.sum(data[istart:istart + config["count"]]["velcor"]) / config["count"]
        correls = np.append(correls, correl)
    return ts, correls

