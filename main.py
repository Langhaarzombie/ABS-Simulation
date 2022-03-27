import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.simulation as simulation

config = {
    "save_file": "abs_simulation.csv",
    "window_size": 3,
    "sphere_count": 10,
    "simulation_timestep": 0.0004,
    "simulation_steps": 1000,
    "simulation_animation_interval": 40,
    "simulation_animation_skip": 3,
    "temperature": 300,
    "sigma": 1,
}

def run():
    """
    Run main simulation.

    Run through the default simulation to generate data files.
    """
    spheres = simulation.initialize(window_size=config["window_size"], count=config["sphere_count"], temperature=config["temperature"], dt=config["simulation_timestep"])

    dt = config["simulation_timestep"]
    with open(config["save_file"], "w") as file:
        for i in np.arange(config["simulation_steps"]):
            spheres = simulation.step(spheres, config["window_size"], config["sigma"], dt, file)

def show():
    """
    Show simulated data.

    Show the animated simulation of generated data files.
    """
    data = np.loadtxt(config["save_file"])
    fig = plt.figure(figsize=(10, 5))
    ax_anim = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    anim = animation.FuncAnimation(fig, _iterate_file, fargs=[data, ax_anim], frames=int(len(data)/(config["sphere_count"]*config["simulation_animation_skip"])), interval=config["simulation_animation_interval"])

    ax2.set_title("Energy")
    exs, epotys, ekinys, etotys = _plot_energy(data)
    ax2.get_xaxis().set_visible(False)
    ax2.plot(exs, epotys, label="Potential")
    ax2.plot(exs, ekinys, label="Kinetic")
    ax2.plot(exs, etotys, label="Total")
    ax2.legend()

    ax3.set_title("Total Momentum")
    mxs, mys = _plot_momentum(data)
    ax3.plot(mxs, mys)

    plt.show()

def _iterate_file(i, data, plot):
    """
    Animation Function for Matplotlib FuncAnimation.

    Create scatterplot for given simulation step.

    Parameters
    ----------
    i: int
        Simulation step.
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    plot: matplotlib.plot.figure
        Figure object for plotting.
    """
    offset = config["simulation_animation_skip"] * config["sphere_count"] * i
    plot.clear()
    plot.set_xlim([0, config["window_size"]])
    plot.set_ylim([0, config["window_size"]])
    plot.set_zlim([0, config["window_size"]])

    xs = data[offset:offset + config["sphere_count"], 0]
    ys = data[offset:offset + config["sphere_count"], 1]
    zs = data[offset:offset + config["sphere_count"], 2]

    # Print simulation step index to compare simulation and plot data
    print(offset / config["sphere_count"])

    plot.scatter(xs, ys, zs)

def _plot_energy(data):
    """
    Calculate total, kinetic and potential energy during simulation for plot.

    Create the plot data to check energy conservation and stability of the system.

    Parameters:
    -----------
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    """
    xs = np.arange(int(len(data)/config["sphere_count"]))
    potential_y = np.array([])
    kinetic_y = np.array([])
    for i in xs:
        istart = config["sphere_count"] * i
        potential_y = np.append(potential_y, np.sum(data[istart:istart + config["sphere_count"], 6]))
        kinetic_y = np.append(kinetic_y, np.sum(data[istart:istart + config["sphere_count"], 7]))

    return xs, potential_y, kinetic_y, kinetic_y + potential_y

def _plot_momentum(data):
    """
    Plot the total momentum energy during simulation.

    Sum over all velocities of spheres and plot them.
    The total momentum should change as little as possible.

    Parameters:
    -----------
    data: numpy.ndarray of numpy.ndarray of int
        List of spheres.
    """
    mom_x = np.arange(int(len(data)/config["sphere_count"]))
    mom_y = np.array([])
    for i in mom_x:
        istart = config["sphere_count"] * i
        mom_y = np.append(mom_y, np.sum(data[istart:istart + config["sphere_count"], 3:5]))

    return mom_x, mom_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="mode", action="store_const", const="show", default="run_show", help="show existing data (default: generate and show new data)")
    args = vars(parser.parse_args())

    if args["mode"] == "run_show":
        run()
    show()
