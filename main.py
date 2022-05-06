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
    ax2 = fig.add_subplot(221)
    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)

    ax2.set_title("Energy")
    ts, epotys, ekinys, etotys = plotter.energy(config, data)
    ax2.plot(ts, epotys, label="Potential")
    ax2.plot(ts, ekinys, label="Kinetic")
    ax2.plot(ts, etotys, label="Total")
    ax2.legend()

    ax3.set_title("Total Momentum")
    _, mxs, mys, mzs = plotter.momentum(config, data)
    ax3.plot(ts, mxs, label="X")
    ax3.plot(ts, mys, label="Y")
    ax3.plot(ts, mzs, label="Z")
    ax3.legend()

    ax4.set_title("Temperature")
    _, temps = plotter.temperature(config, data)
    ax4.plot(ts, temps, label="Red. Temperature")
    ax4.legend()

    ax5.set_title("Radial Distribution")
    r, prob = plotter.radial_distribution(config, data)
    ax5.set_xlabel(r"$r / \sigma$")
    ax5.set_ylabel(r"$g(r)$")
    ax5.axhline(y=1, dashes=(5, 2), color="grey", lw=0.8)
    ax5.plot(r, prob)

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
