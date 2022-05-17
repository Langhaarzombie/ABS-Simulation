import numpy as np

class Sphere:
    """
    Is an Active Brownian Sphere in the simulation.

    Attributes
    ----------
    bounds: int
        Size of observed window, restricts position of spheres.
    position: numpy.ndarray of int
        x, y, z coordinates.
    init_position: numpy.ndarray of int
        x, y, z coordinates at t = 0.
    velocity: numpy.ndarray of int
        Velocity in x, y, z direction.
    init_velocity: numpy.ndarray of int
        Velocity in x, y, z direction at t = 0.
    acceleration: numpy.ndarray of int
        Acceleration in x, y, z direction.
    potential_energy: float64
        Potential energy of sphere.
    """
    def __init__(self, bounds, position, velocity):
        self.bounds = bounds

        self.position = position
        self.velocity = velocity
        self.init_position = self.position
        self.init_velocity = self.velocity

        self.acceleration  = np.zeros(3)
        self.potential_energy = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        """
        Set position of sphere in window.

        Put the sphere at given coordinates inside the
        given window (defined by config). If the given
        coordinates are out of bounds of the window size,
        the sphere enters from the other side of the window
        (periodic boundary).

        Parameters:
        -----------
        value: numpy.ndarray of int
            3d coordinates of new position.
        """
        x, y, z = value
        x = self._correct_boundary(x)
        y = self._correct_boundary(y)
        z = self._correct_boundary(z)

        self._position = np.array([x, y, z])


    def kinetic_energy(self):
        """
        Get kinetic energy of sphere.

        Returns half of of velocity squared.

        Returns:
        --------
        float
            Kinetic energy of the sphere.
        """
        return 0.5*np.dot(self.velocity, self.velocity)

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
        boundary = self.bounds
        if coord < 0:
            return coord + boundary
        elif coord > boundary:
            return coord - boundary
        return coord

    def __str__(self):
        return str(f"""{self.position[0]}\t{self.position[1]}\t{self.position[2]}\t{self.velocity[0]}\t{self.velocity[1]}\t{self.velocity[2]}\t{self.potential_energy}\t{self.kinetic_energy()}""")

