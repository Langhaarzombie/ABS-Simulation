import numpy as np

class Sphere:
    """
    Is an Active Brownian Sphere in the simulation.

    Attributes
    ----------
    location: numpy.ndarray of int
        x, y, z coordinates.
    old_location: numpy.ndarray of int
        x, y, z coordinates of previous location.
    location: numpy.ndarray of int
        x, y, z coordinates.
    velocity: numpy.ndarray of int
        Velocity in x, y, z direction.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.location = np.zeros(3)
        self.old_location  = np.zeros(3)
        self.velocity = np.zeros(3)
        self.force = np.zeros(3)

    def update_sphere(self, new_location, dt):
        """
        Update sphere with new location.

        Simultaneously update location, old_location and
        velocity of a sphere.

        Parameters:
        -----------
        new_location: numpy.ndarray of int
            x, y, z coordinates of new location.
        dt: float
            Timestep size.
        """
        self.velocity = (new_location - self.old_location)/(2*dt)
        self.old_location = self.location
        self.location = new_location

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
        x = self._correct_boundary(x)
        y = self._correct_boundary(y)
        z = self._correct_boundary(z)

        self._location = np.array([x, y, z])

    @property
    def old_location(self):
        return self._old_location

    @old_location.setter
    def old_location(self, value):
        """
        Set the old location of sphere in window.

        Store the old location of the sphere inside the
        given window (defined by config). If the given
        coordinates are out of bounds of the window size,
        the sphere enters from the other side of the window
        (periodic boundary).

        Parameters:
        -----------
        value: numpy.ndarray of int
            3d coordinates of old location.
        """
        x, y, z = value
        x = self._correct_boundary(x)
        y = self._correct_boundary(y)
        z = self._correct_boundary(z)

        self._old_location = np.array([x, y, z])

    def _correct_boundary(self, coord):
        """
        Get the corresponding coordinate value inside the
        window for any coordinate.

        If the given coordinate is outside the window, the
        corresponding coordinate value inside the window is
        returned.

        Parameters:
        -----------
        coord: int
            Single x, y or z coordinate.
        """
        boundary = self.window_size
        if coord < 0:
            return coord + boundary
        elif coord > boundary:
            return coord - boundary
        return coord


    def __str__(self):
        return str(f"""{self.location[0]}\t{self.location[1]}\t{self.location[2]}\t{self.velocity[0]}\t{self.velocity[1]}\t{self.velocity[2]}""")

