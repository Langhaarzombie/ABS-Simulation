import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.simulation as simulation

config = {
    "save_file": "abs_simulation.csv",
    "window_size": 5,
    "sphere_count": 100,
    "simulation_timestep": 0.0001,
    "simulation_steps": 2000,
    "simulation_animation_interval": 40,
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
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    anim = animation.FuncAnimation(fig, _iterate_file, fargs=[data, ax], frames=int(len(data)/config["sphere_count"]), interval=config["simulation_animation_interval"])

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
        Coordinates of spheres.
    plot: matplotlib.plot.figure
        Figure object for plotting.
    """
    offset = config["sphere_count"] * i
    plot.clear()
    plot.set_xlim([0, config["window_size"]])
    plot.set_ylim([0, config["window_size"]])
    plot.set_zlim([0, config["window_size"]])

    xs = data[offset:offset + config["sphere_count"], 0]
    ys = data[offset:offset + config["sphere_count"], 1]
    zs = data[offset:offset + config["sphere_count"], 2]
    ens = data[offset:offset + config["sphere_count"], 6]

    max_en = np.max(ens)
    colors = np.array([])
    for e in ens:
        colors = np.append(colors, e/max_en)

    su = np.sum(ens)
    print(su)

    plot.scatter(xs, ys, zs, c=colors, cmap="coolwarm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="mode", action="store_const", const="show", default="run_show", help="show existing data (default: generate and show new data)")
    args = vars(parser.parse_args())

    if args["mode"] == "run_show":
        run()
    show()
