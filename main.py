import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.stats import maxwell
from src.sphere import Sphere

config = {
    "window_size": 100,
    "sphere_count": 100,
    "simulation_steps": 1000,
    "simulation_stepsize": 1,
    "sphere_mass": 1e-12,
    "temperature": 350
}

def run():
    """
    Run main simulation.

    Run through the default simulation to generate data files.
    """
    print("Running Simulation...")

    # Create spheres
    spheres = np.array([])
    for _ in np.arange(config["sphere_count"]):
        s = Sphere()

        # Distribute sphere in space
        loc_gen = np.random.rand(2)
        s.location = loc_gen * config["window_size"]

        # Assign velocity with Maxwell Boltzmann distribution
        scale = np.sqrt(const.k * config["temperature"] / config["sphere_mass"])
        vel_gen = maxwell.rvs(size=3, scale=scale)
        s.velocity = vel_gen

        print(s.__dict__)
        spheres = np.append(spheres, s)


def show():
    """
    Show simulated data.

    Show the animated simulation of generated data files.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, config["window_size"]])
    ax.set_ylim([0, config["window_size"]])
    ax.set_zlim([0, config["window_size"]])

    ax.scatter([0, 1, 2], [0, 1, 2], [0, 1, 2])

    plt.show()

if __name__ == "__main__":
    run()
    show()
