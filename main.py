import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import src.plotter as plotter
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

    # Run
    saviour = Writer.from_config(config["run"])
    for i in np.arange(config["steps"]):
        spheres = simulation.step(spheres, config["bounds"], config["sigma"], config["timestep"])
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
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)

    ax1.set_title("Energy")
    ts, epotys, ekinys, etotys = plotter.energy(config, data)
    ax1.set_xlabel("Timestep")
    ax1.plot(ts, epotys, label="Potential")
    ax1.plot(ts, ekinys, label="Kinetic")
    ax1.plot(ts, etotys, label="Total")
    ax1.legend()

    ax2.set_title("Total Momentum")
    _, mxs, mys, mzs = plotter.momentum(config, data)
    ax2.set_xlabel("Timestep")
    ax2.plot(ts, mxs, label="X")
    ax2.plot(ts, mys, label="Y")
    ax2.plot(ts, mzs, label="Z")
    ax2.legend()

    ax3.set_title("Temperature")
    _, temps = plotter.temperature(config, data)
    ax3.set_xlabel("Timestep")
    ax3.plot(ts, temps, label="Red. Temperature")
    ax3.legend()

    ax4.set_title("Radial Distribution")
    r, prob = plotter.radial_distribution(config, data)
    ax4.set_xlabel(r"$r / \sigma$")
    ax4.set_ylabel(r"$g(r)$")
    ax4.axhline(y=1, dashes=(5, 2), color="grey", lw=0.8)
    ax4.plot(r, prob)

    ax5.set_title("Velocity Correlation")
    _, velcorr = plotter.velocity_correlation(config, data)
    ax5.set_xlabel("Timestep")
    ax5.set_ylabel(r"$< v(t=0), v(t) >$")
    ax5.axhline(y=0, dashes=(5, 2), color="grey", lw=0.8)
    ax5.plot(ts, velcorr)

    plt.tight_layout()
    plt.show()

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
