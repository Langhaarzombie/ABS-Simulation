import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.simulation as simulation

config = {
    "save_file": "abs_simulation.csv",
    "window_size": 100,
    "sphere_count": 100,
    "simulation_timestep": 0.01,
    "simulation_steps": 1000,
    "simulation_animation_interval": 10,
    "temperature": 50,
    "sigma": 7.5,
    "epsilon": 2
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
            spheres = simulation.step(spheres, config["window_size"], config["sigma"], config["epsilon"], dt, file)

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
    vels = data[offset:offset + config["sphere_count"], 3:6]

    colors = np.array([])
    for vel in vels:
        colors = np.append(colors, np.inner(vel, vel))

    plot.scatter(xs, ys, zs, c=colors)

if __name__ == "__main__":
    run()
    show()
