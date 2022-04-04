import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.simulation as simulation

config = {
    "save_file": "abs_simulation.csv",
    "bounds": 10,
    "sphere_count": 1000,
    "simulation_timestep": 0.0005,
    "simulation_steps": 1000,
    "temperature": 250,
    "sigma": 1,
}

def run():
    """
    Run main simulation.

    Run through the default simulation to generate data files.
    """
    spheres = simulation.initialize(bounds=config["bounds"], count=config["sphere_count"], temperature=config["temperature"], dt=config["simulation_timestep"])

    dt = config["simulation_timestep"]
    with open(config["save_file"], "w") as file:
        for i in np.arange(config["simulation_steps"]):
            spheres = simulation.step(spheres, config["bounds"], config["sigma"], dt, file)

def show():
    """
    Show simulated data.

    Show the animated simulation of generated data files.
    """
    data = np.loadtxt(config["save_file"])
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
        potential = np.append(potential, np.sum(data[istart:istart + config["sphere_count"], 6]))
        kinetic = np.append(kinetic, np.sum(data[istart:istart + config["sphere_count"], 7]))

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
        momentum_x = np.append(momentum_x, np.sum(data[istart:istart + config["sphere_count"], 3]))
        momentum_y = np.append(momentum_y, np.sum(data[istart:istart + config["sphere_count"], 4]))
        momentum_z = np.append(momentum_z, np.sum(data[istart:istart + config["sphere_count"], 5]))
    return ts, momentum_x, momentum_y, momentum_z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="mode", action="store_const", const="show", default="run_show", help="show existing data (default: generate and show new data)")
    args = vars(parser.parse_args())

    if args["mode"] == "run_show":
        run()
    show()
