import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src import initialize, simulation
from src.writer import Writer

config = {
    "init_file": "abs_init.csv",
    "save_file": "abs_simulation.csv",
    "observables": ["position", "velocity", "potential_energy", "kinetic_energy"],
    "bounds": 2,
    "sphere_count": 20,
    "simulation_timestep": 0.0005,
    "simulation_steps": 100,
    "temperature": 250,
    "sigma": 1,
}

def run(init_file, dump_init):
    """
    Run main simulation.

    Run through the default simulation to generate data files.

    Parameters:
    -----------
    init_file: str
        File name with init configuration.
    dump_init:
        File to store init configuration to.
    """
    if init_file:
        spheres = initialize.from_file(init_file)
    else:
        spheres = initialize.random(bounds=config["bounds"], count=config["sphere_count"], temperature=config["temperature"], dt=config["simulation_timestep"])

    if dump_init:
        saviour = Writer(dump_init, ["position", "velocity", "bounds"])
        saviour.write(spheres)
        saviour.close_file()

    dt = config["simulation_timestep"]
    saviour = Writer(config["save_file"], config["observables"])
    for i in np.arange(config["simulation_steps"]):
        spheres = simulation.step(spheres, config["bounds"], config["sigma"], dt)
        saviour.write(spheres)
    saviour.close_file()

def show():
    """
    Show simulated data.

    Show the animated simulation of generated data files.
    """
    data = np.genfromtxt(config["save_file"], names=True)
    fig = plt.figure(figsize=(10, 5))
    ax2 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212)

    ax2.set_title("Energy")
    exs, epotys, ekinys, etotys = _plot_energy(data)
    ax2.get_xaxis().set_visible(False)
    ax2.plot(exs, epotys, label="Potential")
    ax2.plot(exs, ekinys, label="Kinetic")
    ax2.plot(exs, etotys, label="Total")
    ax2.legend()

    ax3.set_title("Total Momentum")
    mts, mxs, mys, mzs = _plot_momentum(data)
    ax3.plot(mts, mxs, label="X")
    ax3.plot(mts, mys, label="Y")
    ax3.plot(mts, mzs, label="Z")
    ax3.legend()

    plt.show()

def _plot_energy(data):
    """
    Calculate total, kinetic and potential energy during simulation for plot.

    Create the plot data to check energy conservation and stability of the system.

    Parameters:
    -----------
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    """
    ts = np.arange(int(len(data)/config["sphere_count"]))
    potential = np.array([])
    kinetic = np.array([])
    for i in ts:
        istart = config["sphere_count"] * i
        p = np.sum(data[istart:istart + config["sphere_count"]]["pen"])
        k = np.sum(data[istart:istart + config["sphere_count"]]["ken"])
        potential = np.append(potential, p)
        kinetic = np.append(kinetic, k)

    total_energy = kinetic + potential

    print(f"(std / mean) of total energy: {np.std(total_energy) / np.mean(total_energy)}")

    return ts, potential, kinetic, total_energy

def _plot_momentum(data):
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
    ts = np.arange(int(len(data)/config["sphere_count"]))
    momentum_x = np.array([])
    momentum_y = np.array([])
    momentum_z = np.array([])
    for i in ts:
        istart = config["sphere_count"] * i
        momentum_x = np.append(momentum_x, np.sum(data[istart:istart + config["sphere_count"]]["vx"]))
        momentum_y = np.append(momentum_y, np.sum(data[istart:istart + config["sphere_count"]]["vy"]))
        momentum_z = np.append(momentum_z, np.sum(data[istart:istart + config["sphere_count"]]["vz"]))
    return ts, momentum_x, momentum_y, momentum_z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="mode", action="store_const", const="show", default="run_show", help="show existing data (default: generate and show new data)")
    parser.add_argument("-i", "--init", dest="init", action="store", default=None, help="use init file for simulation (default: init randomly)")
    parser.add_argument("-d", "--dump-init", dest="dump_init", action="store", default=None, help="dump init config into file (default: don\'t save init)")
    args = vars(parser.parse_args())

    if args["mode"] == "run_show":
        run(init_file=args["init"], dump_init=args["dump_init"])
    show()
