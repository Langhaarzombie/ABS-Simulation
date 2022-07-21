import numpy as np
from src.sphere import Sphere

class Writer:
    """
    Writes save data of simulation to given file.

    Extracts given observables of ABS in simulation
    and writes them to given save files.

    Attributes:
    -----------
    filenames: numpy.array of str
        Save file to write to.
    csv_observables: numpy.array of str
        Observables of simulation to save to csv file.
    header_names:
        Array to map attribute names and observables to headers in csv.
    """
    header_names = {
        "position": "x\ty\tz",
        "velocity": "vx\tvy\tvz",
        "kinetic_energy": "ken",
        "potential_energy": "pen",
        "active_acceleration": "ax\tay\taz",
        "temperature": "temp",
        "bounds": "b",
        "velocity_correlation": "velcor",
        "nearest_neighbours": "nn"
    }

    def __init__(self, config, csv_observables):
        self.csv_observables = csv_observables
        self.skip = config["run"]["write_skips"]
        self.counter = self.skip
        self.files = np.array([])
        for f in config["run"]["files"]:
            newfile = open(f, "w")
            self.write_header(config, newfile)
            self.files = np.append(self.files, newfile)

    @classmethod
    def from_config(cls, config):
        """
        Create Writer object from config file.

        Parameters:
        -----------
        config: dict
            Config read from yaml config file.
        Returns:
        --------
        Writer
            Writer object to save data with.
        """
        # TODO implement that writer considers how often to calc observables
        observables = [*config["run"]["csv_observables"].keys()]
        return cls(config, observables)

    def close_file(self):
        """
        Close the files the simulation is stored in.

        Parameters:
        -----------
        file: _io.TextIOWrapper
            Opened file.
        """
        for f in self.files:
            f.close()

    def write(self, spheres):
        """
        Write data of simulation to specified file and format
        if the skip counter allows it.

        Parameters:
        -----------
        spheres: numpy.array of Spheres
            Array of ABS in simulation.
        """
        if self.counter == self.skip:
            self.counter = 1
            for f in self.files:
                if f.name[-4:] == ".csv":
                    self.write_csv(spheres, f)
                elif f.name[-4:] == ".xyz":
                    self.write_xyz(spheres, f)
        else:
            self.counter += 1

    def write_xyz(self, spheres, file):
        """
        Write XYZ data of simulation step to save file.

        Parameters:
        -----------
        spheres: numpy.array of Spheres
            Array of ABS in simulation.
        file: _io.TextIOWrapper
            Openend csv file to write to.
        """
        for i, s in enumerate(spheres):
            data_line = f"s{i}\t" + self.get_observable(spheres, s, "position") + "\t"
            print(data_line, end="\n", file=file)

    def write_csv(self, spheres, file):
        """
        Write CSV data of simulation step to save file.

        Parameters:
        -----------
        spheres: numpy.array of Spheres
            Array of ABS in simulation.
        file: _io.TextIOWrapper
            Openend csv file to write to.
        """
        for s in spheres:
            data_line = ""
            for o in self.csv_observables:
                data_line += self.get_observable(spheres, s, o) + "\t"
            print(data_line, end="\n", file=file)

    def write_header(self, config, file):
        """
        Write header for specified file to file.

        Parameters:
        -----------
        config: dict
            Config of simulation.
        file: _io.TextIOWrapper
            Openend csv file to write to.
        """
        if file.name[-4:] == ".csv":
            print(self.get_csv_header(config), end="\n", file=file)
        elif file.name[-4:] == ".xyz":
            print(self.get_xyz_header(config), end="\n", file=file)

    def get_xyz_header(self, config):
        """
        Get header string to print at top of xyz file.

        Builds the string that will be printed at the
        top of the xyz file. Includes the count of spheres
        and information about the simulations config.

        Parameters:
        -----------
        config: dict
            Config of simulation.
        Returns:
        --------
        str
            Header string for xyz file.
        """
        return str(config["count"]) + "\nDensity: " + str(config["density"]) + ", Temperature: " + str(config["temperature"]) + ", Steps: " + str(config["steps"])

    def get_csv_header(self, config):
        """
        Get header string to print at top of csv file.

        Builds the string that will be printed at the
        top of the csv file. Includes the names of the
        observables defined at object construction.

        Returns:
        --------
        str
            Header string for csv file.
        Raises:
        -------
        KeyError:
            If an observable is not defined.
        """
        header = ""
        for o in self.csv_observables:
            try:
                header += Writer.header_names[o] + "\t"
            except KeyError:
                print(f"No header defined for observable of name: {o}")
        return header[:-1]

    def get_observable(self, spheres, sphere, observable):
        """
        Get data from sphere associated with observable name.

        Returns the stringified data of a sphere object that is
        associated with the given observable name.
        Throws an excpetion if the observable is not defined.

        Parameters:
        -----------
        spheres: numpy.array of Sphere
            Array of spheres used in simulation.
        sphere: Sphere
            Sphere to get data from.
        observable: str
            Name of requested observable.
        Returns:
        --------
        str
            Data string of observable.
        Raises:
        -------
        KeyError:
            If given observable is not defined.
        """
        if observable == "position":
            return f"{sphere.position[0]}\t{sphere.position[1]}\t{sphere.position[2]}"
        elif observable == "velocity":
            return f"{sphere.velocity[0]}\t{sphere.velocity[1]}\t{sphere.velocity[2]}"
        elif observable == "kinetic_energy":
            return f"{sphere.kinetic_energy()}"
        elif observable == "potential_energy":
            return f"{sphere.potential_energy}"
        elif observable == "active_acceleration":
            return f"{sphere.active_acceleration[0]}\t{sphere.active_acceleration[1]}\t{sphere.active_acceleration[2]}"
        elif observable == "temperature":
            return f"{2*sphere.kinetic_energy() / (3 * len(spheres))}"
        elif observable == "bounds":
            return f"{sphere.bounds}"
        elif observable == "velocity_correlation":
            return f"{np.dot(sphere.init_velocity, sphere.velocity) / (np.linalg.norm(sphere.init_velocity) * np.linalg.norm(sphere.velocity))}"
        elif observable == "nearest_neighbours":
            return f"{sphere.average_nearest_neighbour}"
        else:
            raise KeyError(f"Undefined observable requested: {observable}")
