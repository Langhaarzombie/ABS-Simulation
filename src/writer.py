import numpy as np
from src.sphere import Sphere

class Writer:
    """
    Writes save data of simulation to given file.

    Extracts given observables of ABS in simulation
    and writes them to a given save file.

    Attributes:
    -----------
    file: _io.TextIOWrapper
        Save file to write to.
    observables: numpy.array of str
        Observables of simulation to save.
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
        "velocity_correlation": "velcor"
    }

    def __init__(self, filename, observables):
        self.file = open(filename, "w")
        self.observables = observables
        print(self.get_header(), end="\n", file=self.file)

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
        filename = config["save_file"]
        # TODO implement that writer considers how often to calc observables
        observables = [*config["observables"].keys()]
        return cls(filename, observables)

    def close_file(self):
        """
        Close the file the simulation is stored in.

        Parameters:
        -----------
        file: _io.TextIOWrapper
            Opened file.
        """
        self.file.close()

    def write(self, spheres):
        """
        Write data of simulation step to save file.

        Parameters:
        -----------
        file: _io.TextIOWrapper
            Opened file.
        spheres: numpy.array of Spheres
            Array of ABS in simulation.
        """
        for s in spheres:
            data_line = ""
            for o in self.observables:
                data_line += self.get_observable(spheres, s, o) + "\t"
            print(data_line, end="\n", file=self.file)

    def get_header(self):
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
        for o in self.observables:
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
        else:
            raise KeyError(f"Undefined observable requested: {observable}")
