import numpy as np

class Sphere:
    """
    Is an Active Brownian Sphere in the simulation.

    Attributes
    ----------
    location: numpy.ndarray of int
        x, y, z coordinates.
    velocity: numpy.ndarray of int
        Velocity in x, y, z direction.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.location = np.array([0, 0, 0])
        self.velocity = np.array([0, 0, 0])

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        """
        Set location of sphere in window.

        Put the sphere at given coordinates inside the
        given window (defined by config). If the given
        coordinates are out of bounds of the window size,
        the sphere enters from the other side of the window
        (periodic boundary).

        Parameters:
        -----------
        value: numpy.ndarray of int
            3d coordinates of new location.
        """
        x, y, z = value
        boundary = self.window_size

        if x < 0:
            x = x + boundary
        elif x > boundary:
            x = x - boundary

        if y < 0:
            y = y + boundary
        elif y > boundary:
            y = y - boundary

        if z < 0:
            z = z + boundary
        elif z > boundary:
            z = z - boundary

        self._location = np.array([x, y, z])

    def __str__(self):
        return str(f"{self.location[0]}\t{self.location[1]}\t{self.location[2]}\t{self.velocity[0]}\t{self.velocity[1]}\t{self.velocity[2]}")

