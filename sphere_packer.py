from __future__ import division
from collections import defaultdict
from math import pi, sin, cos
import numpy as np
import itertools
import random
from random import uniform, gauss

class RandomSequentialPack(object):
    """Random sequential sphere packer. Spheres are placed one by one at
    rondom inside the domain. Placement attempts are made until the sphere is
    not overlapping any others. This algorithm uses a lattice over the domain
    to speed up nearest neighbor search by only searching for a sphere's
    neighbors within that lattice cell.

    Parameters
    ----------
    radius : float
        Radius of spheres
    geometry : {'cube', 'cylinder', or 'sphere'}
        Geometry of the domain
    domain_length : float
        Length of domain (if cube or cylinder)
    domain_radius : float
        Radius of domain (if cylinder or sphere)
    spheres : int
        Number of spheres to pack in domain. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    packing_fraction : float
        Packing fraction of spheres. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    lattice_dimension : array_like, optional
        Number of lattice cells in each dimension
    seed : int, optional
        RNG seed

    Attributes
    ----------
    radius : float
        Radius of spheres
    geometry : {'cube', 'cylinder', or 'sphere'}
        Geometry of the domain
    domain_length : float
        Length of domain (if cube or cylinder)
    domain_radius : float
        Radius of domain (if cylinder or sphere)
    spheres : int
        Number of spheres to pack in domain. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    packing_fraction : float
        Packing fraction of spheres. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    lattice_dimension : numpy.ndarray
        Number of lattice cells in each dimension
    seed : int
        RNG seed
    cell_length : numpy.ndarray
        Length of lattice cells in each dimension
    sphere_volume : float
        volume of each sphere
    domain_volume : float
        volume of domain

    """

    def __init__(self, radius, geometry='cylinder', domain_length=None,
                 domain_radius=None, spheres=None, packing_fraction=None,
                 lattice_dimension=None, seed=1):
        # Initialize RandomSequentialPacker class attributes
        self._spheres = None
        self._packing_fraction = None
        self._radius = None
        self._geometry = None
        self._domain_length = None
        self._domain_radius = None
        self._lattice_dimension = None
        self._seed = None
        self._cell_length = None

        # Set attributes
        self.radius = radius
        self.geometry = geometry
        self.domain_length = domain_length
        self.domain_radius = domain_radius
        if ((spheres is None and packing_fraction is None) or
           (spheres is not None and packing_fraction is not None)):
            raise ValueError("Exactly one of 'spheres' and 'packing_fraction' "
                             "must be specified.")
        if spheres is not None:
            self.spheres = spheres
        if packing_fraction is not None:
            self.packing_fraction = packing_fraction
        if lattice_dimension is None:
            n = int(self.domain_length/(4*self.radius))
            m = int(self.domain_radius/(2*self.radius))
            if self.geometry is 'cube':
                self.lattice_dimension = [n, n, n]
            elif self.geometry is 'cylinder':
                self.lattice_dimension = [m, m, n]
            elif self.geometry is 'sphere':
                self.lattice_dimension = [m, m, m]
        else:
            self.lattice_dimension = lattice_dimension
        self.seed = seed
        if self.geometry is 'cube':
            self.cell_length = [self.domain_length/i for i in
                                self.lattice_dimension]
        elif self.geometry is 'cylinder':
            self.cell_length = [
                2*self.domain_radius/self.lattice_dimension[0],
                2*self.domain_radius/self.lattice_dimension[1],
                self.domain_length/self.lattice_dimension[2]]
        elif self.geometry is 'sphere':
            self.cell_length = [2*self.domain_radius/i for i in
                                self.lattice_dimension]

    @property
    def radius(self):
        return self._radius

    @property
    def geometry(self):
        return self._geometry

    @property
    def domain_length(self):
        return self._domain_length

    @property
    def domain_radius(self):
        return self._domain_radius

    @property
    def spheres(self):
        return self._spheres

    @property
    def packing_fraction(self):
        return self._packing_fraction

    @property
    def lattice_dimension(self):
        return self._lattice_dimension

    @property
    def seed(self):
        return self._seed

    @property
    def cell_length(self):
        return self._cell_length

    @property
    def sphere_volume(self):
        return 4/3 * pi * self.radius**3

    @property
    def domain_volume(self):
        if self.geometry is 'cube':
            return self.domain_length**3
        elif self.geometry is 'cylinder':
            return (self.domain_length * pi * self.domain_radius**2)
        elif self.geometry is 'sphere':
            return 4/3 * pi * self.domain_radius**3

    @radius.setter
    def radius(self, radius):
        if radius <= 0:
            raise ValueError('Sphere radius must be positive value.')
        self._radius = radius

    @geometry.setter
    def geometry(self, geometry):
        if geometry not in ['cube', 'cylinder', 'sphere']:
            raise ValueError('Unable to set geometry to "{}". Only "cube", '
                             '"cylinder", and "sphere" are '
                             'supported."'.format(geometry))
        self._geometry = geometry

    @domain_length.setter
    def domain_length(self, domain_length):
        if domain_length <= 0 and self.geometry in ['cube', 'cylinder']:
            raise ValueError('Domain length must be positive value.')
        self._domain_length = domain_length

    @domain_radius.setter
    def domain_radius(self, domain_radius):
        if domain_radius <= 0 and self.geometry in ['cylinder', 'sphere']:
            raise ValueError('Domain radius must be positive value.')
        self._domain_radius = domain_radius

    @spheres.setter
    def spheres(self, spheres):
        if spheres < 0:
            raise ValueError('Unable to set "spheres" to {}: number of '
                             'spheres must be positive.'.format(spheres))
        packing_fraction = self.sphere_volume * spheres / self.domain_volume
        if packing_fraction < 0:
            raise ValueError('Unable to set packing fraction to {}: packing '
                             'fraction must be '
                             'positive.'.format(packing_fraction))
        if packing_fraction >= 0.38:
            raise ValueError('Packing fraction of {} is greater than the '
                             'packing fraction limit for random sequential '
                             'packing (0.38)'.format(packing_fraction))
        self._spheres = int(spheres)
        self._packing_fraction = packing_fraction

    @packing_fraction.setter
    def packing_fraction(self, packing_fraction):
        if packing_fraction < 0:
            raise ValueError('Unable to set packing fraction to {}: packing '
                             'fraction must be ' \
                             'positive.'.format(packing_fraction))
        if packing_fraction >= 0.38:
            raise ValueError('Packing fraction of {} is greater than the '
                             'packing fraction limit for random sequential '
                             'packing (0.38)'.format(packing_fraction))
        spheres = packing_fraction * self.domain_volume // self.sphere_volume
        if spheres < 0:
            raise ValueError('Unable to set "spheres" to {}: number of '
                             'spheres must be positive.'.format(spheres))
        self.spheres = spheres

    @lattice_dimension.setter
    def lattice_dimension(self, lattice_dimension):
        d = np.asarray(lattice_dimension)
        if d.size is not 3:
            raise ValueError('Unable to set lattice dimension of size {}: '
                             'must be size 3 array specifying number of '
                             'cells in x, y, and z dimensions of the '
                             'lattice'.format(d.size))
        if any(n < 0 for n in d):
            raise ValueError('Lattice dimensions must be positive values.')
        msg = ('Length of lattice cells must be smaller than 2 x sphere '
	       'diameter. Reduce either lattice dimension or sphere radius.')
        if self.geometry is 'cube' and (
            any(self.domain_length/n < 4*self.radius for n in d)):
                raise ValueError(msg)
        elif self.geometry is 'cylinder' and (
            any(self.domain_radius/n < 2*self.radius for n in d[0:1]) or
            self.domain_length/d[2] < 4*self.radius):
                raise ValueError(msg)
        elif self.geometry is 'sphere' and (
            any(self.domain_radius/n < 2*self.radius for n in d)):
                raise ValueError(msg)
        self._lattice_dimension = [i for i in d]

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @cell_length.setter
    def cell_length(self, cell_length):
        self._cell_length = cell_length


    def _random_point_cube(self):
        """Generate Cartesian coordinates of sphere center of a sphere that is
        contained entirely within cubic domain with uniform probability.

        Returns
        -------
        coordinates : ndarray
            Cartesian coordinates of sphere center

        """

        a, b = self.radius, self.domain_length - self.radius

        return [uniform(a, b), uniform(a, b), uniform(a, b)]


    def _random_point_cylinder(self):
        """Generate Cartesian coordinates of sphere center of a sphere that is
        contained entirely within cylindrical domain with uniform probability.
        See http://mathworld.wolfram.com/DiskPointPicking.html for generating
        random points on a disk

        Returns
        -------
        coordinates : ndarray
            Cartesian coordinates of sphere center

        """
        r = uniform(0, (self.domain_radius - self.radius)**2)**.5
        t = uniform(0, 2*pi)

        return [r*cos(t), r*sin(t), uniform(self.radius,
                self.domain_length - self.radius)]


    def _random_point_sphere(self):
        """Generate Cartesian coordinates of sphere center of a sphere that is
        contained entirely within spherical domain with uniform probability.

        Returns
        -------
        coordinates : ndarray
            Cartesian coordinates of sphere center

        """

        x = (gauss(0, 1), gauss(0, 1), gauss(0, 1))
	r = (uniform(0, (self.domain_radius - self.radius)**3)**(1/3) /
             (x[0]**2 + x[1]**2 + x[2]**2)**.5)

        return [r*i for i in x]


    def _cell_index_cube(self, point):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center

        Returns
        -------
        index : list
            indices of lattice cell

        """

        return [int(point[i]/self.cell_length[i]) for i in range(3)]


    def _cell_index_cylinder(self, point):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center

        Returns
        -------
        index : list
            indices of lattice cell

        """

        return [int((point[0] + self.domain_radius)/self.cell_length[0]),
                int((point[1] + self.domain_radius)/self.cell_length[1]),
                int(point[2]/self.cell_length[2])]


    def _cell_index_sphere(self, point):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center

        Returns
        -------
        index : list
            indices of lattice cell

        """

        return [int((point[i] + self.domain_radius)/self.cell_length[i])
                for i in range(3)]


    def _cell_list_cube(self, point, distance):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center
        distance : float
            Find all lattice cells that are within a radius of length
            'distance' from the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

        r = [[i/self.cell_length[j] for i in [point[j] - distance, point[j],
             point[j] + distance] if i > 0 and i < self.domain_length]
             for j in range(3)]

        return list(itertools.product(*({int(i) for i in j} for j in r)))


    def _cell_list_cylinder(self, point, distance):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center
        distance : float
            Find all lattice cells that are within a radius of length
            'distance' from the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

        x,y = [[(i + self.domain_radius)/self.cell_length[j] for i in 
               [point[j] - distance, point[j], point[j] + distance]
               if i > -self.domain_radius and i < self.domain_radius]
               for j in range(2)]

        z = [i/self.cell_length[2] for i in [point[2] - distance, point[2],
             point[2] + distance] if i > 0 and i < self.domain_length]

        return list(itertools.product(*({int(i) for i in j} for j in (x,y,z))))


    def _cell_list_sphere(self, point, distance):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        point : ndarray
            Cartesian coordinates of sphere center
        distance : float
            Find all lattice cells that are within a radius of length
            'distance' from the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

        r = [[(i + self.domain_radius)/self.cell_length[j] for i in
             [point[j] - distance, point[j], point[j] + distance] 
             if i > -self.domain_radius and i < self.domain_radius]
             for j in range(3)]

        return list(itertools.product(*({int(i) for i in j} for j in r)))


    def pack(self):
        """Randomly place all spheres in domain such that they are not
        overlapping

        Returns
        -------
        spheres : numpy.ndarray
            Cartesian coordinates of all spheres in the domain.

        """

        random.seed(self.seed)

        diameter = 2*self.radius
        sqdiameter = diameter**2

        # Set domain dependent functions
        if self.geometry is 'cube':
            random_point = self._random_point_cube
            cell_index = self._cell_index_cube
            cell_list = self._cell_list_cube
        elif self.geometry is 'cylinder':
            random_point = self._random_point_cylinder
            cell_index = self._cell_index_cylinder
            cell_list = self._cell_list_cylinder
        elif self.geometry is 'sphere':
            random_point = self._random_point_sphere
            cell_index = self._cell_index_sphere
            cell_list = self._cell_list_sphere

        #spheres = np.zeros((self.spheres, 3))
        spheres = []
        mesh = defaultdict(list)

        for s in range(self.spheres):

            # Randomly choose position of sphere center within the domain and
            # continue sampling new center coordinates as long as there are any
            # overlaps
            while True:
               p = random_point()
               i,j,k = cell_index(p)
               if any((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2 < sqdiameter
                      for q in mesh[i,j,k]):
                   continue
               else:
                   break
            #spheres[s] = p
            spheres.append(p)

            # Determine which lattice cells are within one diameter of sphere's
            # center and add this sphere to the list of spheres in those cells
            for i,j,k in cell_list(p, diameter):
                mesh[i,j,k].append(p)

        return spheres
