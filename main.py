import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src import initialize, simulation
from src.writer import Writer

def run(config, init_file):
    """
    Run main simulation.

    Run through the default simulation to generate data files.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    init_file:
        If defined, use this file as init data.
    """
    # Initialize
    if init_file:
        spheres = initialize.from_file(init_file)
        config["count"] = len(spheres)
    else:
        spheres = initialize.random(bounds=config["bounds"], count=config["count"], temperature=config["temperature"], dt=config["timestep"], save_file=config["init"]["save_file"])

    dt = config["timestep"]
    saviour = Writer.from_config(config["run"])
    for i in np.arange(config["steps"]):
        spheres = simulation.step(spheres, config["bounds"], config["sigma"], dt)
        saviour.write(spheres)
    saviour.close_file()

def show(config, data_file):
    """
    Show simulated data.

    Show the plots of generated simulation data.

    Parameters:
    -----------
    config: dict
        Config of simulation.
    data_file:
        Filename of simulation data.
    """
    data = np.genfromtxt(data_file, names=True)
    fig = plt.figure(figsize=(10, 5))
    ax2 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212)

    ax2.set_title("Energy")
    exs, epotys, ekinys, etotys = _plot_energy(config, data)
    ax2.get_xaxis().set_visible(False)
    ax2.plot(exs, epotys, label="Potential")
    ax2.plot(exs, ekinys, label="Kinetic")
    ax2.plot(exs, etotys, label="Total")
    ax2.legend()

    ax3.set_title("Total Momentum")
    mts, mxs, mys, mzs = _plot_momentum(config, data)
    ax3.plot(mts, mxs, label="X")
    ax3.plot(mts, mys, label="Y")
    ax3.plot(mts, mzs, label="Z")
    ax3.legend()

    plt.show()

def _plot_energy(config, data):
    """
    Calculate total, kinetic and potential energy during simulation for plot.

    Create the plot data to check energy conservation and stability of the system.

    Parameters:
    -----------
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    """
    ts = np.arange(int(len(data)/config["count"]))
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

def _plot_momentum(config, data):
    """
    Get momenta behaviour over time.

    Gives the time and momenta of center of mass in x, y, z direction
    over time of simulation.

    Parameters:
    -----------
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
    ts = np.arange(int(len(data)/config["count"]))
    momentum_x = np.array([])
    momentum_y = np.array([])
    momentum_z = np.array([])
    for i in ts:
        istart = config["count"] * i
        momentum_x = np.append(momentum_x, np.sum(data[istart:istart + config["count"]]["vx"]))
        momentum_y = np.append(momentum_y, np.sum(data[istart:istart + config["count"]]["vy"]))
        momentum_z = np.append(momentum_z, np.sum(data[istart:istart + config["count"]]["vz"]))
    return ts, momentum_x, momentum_y, momentum_z

def _load_config(filename):
    """
    Get init, run, show config from yaml file.

    Read config for init, run, show from given file.

    Parameters:
    -----------
    filename: str
        Name of yaml config file.
    Returns:
    --------
    dict
        Config for running the simulation.
    """
    with open(filename, "r") as f:
        content = yaml.safe_load(f)
    return content["abs"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="show_file", action="store", default=None, help="show plots of simulation data in file (default: newly generate data)")
    parser.add_argument("-i", "--init", dest="init_file", action="store", default=None, help="use init file for simulation (default: init randomly)")
    parser.add_argument("-c", "--config", dest="config_file", action="store", default="abs_config.yml", help="path to config file (default: abs_config.yml)")
    args = vars(parser.parse_args())

    config = _load_config(args["config_file"])

    if not args["show_file"]:
        run(config, args["init_file"])
        show(config, config["run"]["save_file"])
    else:
        show(config, args["show_file"])
