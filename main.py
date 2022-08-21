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
        config["bounds"] = np.cbrt(len(spheres) / config["density"])
    else:
        spheres, config["bounds"] = initialize.random(config)

    # Run
    equilibration_time = 480000
    equilibration_time = 0
    saviour = Writer.from_config(config)
    for i in np.arange(config["steps"]):
        spheres = simulation.step(spheres, config["bounds"], config["sigma"], config["temperature"], config["timestep"])
        if equilibration_time <= 0:
            print(f"{i}", end=",")
            saviour.write(spheres)
        else:
            equilibration_time -= config["run"]["write_skips"]
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
        Filename of csv simulation data.
    """
    #  data = np.genfromtxt(data_file, names=True)
    #  data = _load_data(data_file)[500000:]
    data = _load_data(data_file)
    print(data)
    fig = plt.figure(figsize=(10, 5))
    #  ax1 = fig.add_subplot(231)
    #  ax2 = fig.add_subplot(232)
    #  ax3 = fig.add_subplot(233)
    #  ax4 = fig.add_subplot(234)
    #  ax5 = fig.add_subplot(235)
    ax4 = fig.add_subplot(111)

    #  ax1.set_title("Energy")
    #  ts, epotys, ekinys, etotys = plotter.energy(config, data)
    #  ax1.set_xlabel("Timestep")
    #  ax1.set_ylabel(r"$\epsilon$")
    #  ax1.plot(ts, epotys, label="Potential")
    #  ax1.plot(ts, ekinys, label="Kinetic")
    #  ax1.plot(ts, etotys, label="Total")
    #  ax1.legend()

    #  ax2.set_title("Total Momentum")
    #  _, mxs, mys, mzs = plotter.momentum(config, data)
    #  ax2.set_xlabel("Timestep")
    #  ax3.set_ylabel("p*")
    #  ax2.plot(ts, mxs, label="X")
    #  ax2.plot(ts, mys, label="Y")
    #  ax2.plot(ts, mzs, label="Z")
    #  ax2.legend()

    #  ax3.set_title("Temperature")
    #  ts, temps = plotter.temperature(config, data)
    #  ax3.set_xlabel("Timestep")
    #  ax3.set_ylabel("T*")
    #  ax3.plot(ts + 480000, temps, label="Red. Temperature")
    #  ax3.legend()

    ax4.set_title("Radial Distribution")
    r, prob = plotter.radial_distribution(config, data)
    ax4.set_xlabel(r"$r / \sigma$")
    ax4.set_ylabel(r"$g(r)$")
    ax4.axhline(y=1, dashes=(5, 2), color="grey", lw=0.8)
    ax4.plot(r, prob)

    #  ax5.set_title("Velocity Correlation")
    #  ls, velcorr = plotter.velocity_correlation(config, data)
    #  ax5.set_xlabel("l")
    #  ax5.set_ylabel(r"$< v(t=k), v(t=k+l) >$")
    #  ax5.axhline(y=0, dashes=(5, 2), color="grey", lw=0.8)
    #  ax5.plot(ls, velcorr)

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

def _load_data(filename, delimiter="\t", skiprows=1, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        _load_data.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, _load_data.rowlength))
    data.dtype = {"names": ["x", "y", "z", "vx", "vy", "vz", "pen", "ken", "temp", "nn"], "formats": [float, float, float, float, float, float, float, float, float, float]} # TODO only a quick fix as it assumes all obserrvables
    return data

def _print_simulation_info(filename, config):
    """
    Print configuration of running simulation at start.

    Parameters:
    -----------
    filename: str
        Name of yaml config file.
    config: dict
        Config for running the simulation.
    """
    # TODO store config at top of csv file, to show correct config when
    # only showing data
    print(f"MD simulation with config file: {filename}")
    for k in config.keys():
        print(f"{k}:  {config[k]}")
    bounds = np.cbrt(config["count"] / config["density"])
    print(f"bounds:  {bounds}")
    volfrac = config["count"]*np.pi*(2**(1/6)*config["sigma"])**3/(6*bounds**3)
    print(f"vol.frac.:  {volfrac}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", dest="show_file", action="store", default=None, help="show plots of simulation data in file (default: newly generate data)")
    parser.add_argument("-i", "--init", dest="init_file", action="store", default=None, help="use init file for simulation (default: init randomly on fcc)")
    parser.add_argument("-c", "--config", dest="config_file", action="store", default="abs_config.yml", help="path to config file (default: abs_config.yml)")
    args = vars(parser.parse_args())

    config = _load_config(args["config_file"])
    _print_simulation_info(args["config_file"], config)

    if not args["show_file"]:
        run(config, args["init_file"])
        for f in config["run"]["files"]:
            if f[-4:] == ".csv":
                #  show(config, f)
                pass
    else:
        show(config, args["show_file"])
