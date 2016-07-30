from __future__ import division
import numpy as np
import itertools
import bisect
import scipy.spatial
from scipy.spatial.distance import cdist
from collections import defaultdict
from math import pi, floor, log10

class CloseRandomPack(object):
    """Close random sphere packer using Jodrey and Tory's algorithm. At the
    beginning of the simulation all spheres are placed randomly within the
    domain. Each sphere has two diameters, and inner and an outer, which
    approach each other during the simulation. The inner diameter, defined as
    the minimum center-to-center distance, is the true diameter of the spheres
    and defines the packing fraction. At each iteration the worst overlap
    between spheres based on outer diameter is eliminated by moving the spheres
    apart along the line joining their centers and the outer diameter is
    decreased. Iterations continue until the two diameters converge or until
    the desired packing fraction is reached.

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
    n_spheres : int
        Number of spheres to pack in domain. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    packing_fraction : float
        Packing fraction of spheres. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    contraction_rate : float
        Contraction rate of outer diameter
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
    n_spheres : int
        Number of spheres to pack in domain. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    packing_fraction : float
        Final desired packing fraction of spheres. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    contraction_rate : float
        Contraction rate of outer diameter
    seed : int
        RNG seed
    sphere_volume : float
        Volume of each sphere
    domain_volume : float
        Volume of domain
    inner_diameter : float
        Inner diameter of spheres, defined as minimum center-to-center distance
        between any two spheres
    outer_diameter : float
        Outer diameter of spheres
    initial_outer_diameter : float
        Initial outer diameter of spheres which is set to an arbitrary value
        such that packing fraction is 1
    inner_packing_fraction : float
        True packing fraction of spheres calculated from inner diameter.
    outer_packing_fraction : float
        Nominal packing fraction of spheres calculated from outer diameter.
    spheres : numpy.ndarray, shape = [n_spheres, 3]
        Cartesian coordinates of sphere centers
    rods : list, shape = [n_rods, 3]
        Sorted list of rods where each element contains the distance between
        the sphere centers and the indices of the two spheres

    """

    def __init__(self, radius, geometry='cylinder', domain_length=None,
                 domain_radius=None, n_spheres=None, packing_fraction=None,
                 contraction_rate=None, seed=1):
        # Initialize RandomSequentialPacker class attributes
        self._n_spheres = None
        self._packing_fraction = None
        self._radius = None
        self._geometry = None
        self._domain_length = None
        self._domain_radius = None
        self._contraction_rate = None
        self._seed = None
        self._inner_diameter = None
        self._outer_diameter= None

        # Set attributes
        self.radius = radius
        self.geometry = geometry
        self.domain_length = domain_length
        self.domain_radius = domain_radius
        if ((n_spheres is None and packing_fraction is None) or
           (n_spheres is not None and packing_fraction is not None)):
            raise ValueError("Exactly one of 'n_spheres' and 'packing_fraction' "
                             "must be specified.")
        if n_spheres is not None:
            self.n_spheres = n_spheres
        if packing_fraction is not None:
            self.packing_fraction = packing_fraction
        if contraction_rate is not None:
            self.contraction_rate = contraction_rate
        else:
            self.contraction_rate = 1
        self.seed = seed
        self.initial_outer_diameter = 2 * (self.domain_volume / 
                                          (self.n_spheres * 4/3 * pi))**(1/3)
        self.outer_diameter = self.initial_outer_diameter
        self.spheres = None
        self.rods = None

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
    def n_spheres(self):
        return self._n_spheres

    @property
    def packing_fraction(self):
        return self._packing_fraction

    @property
    def contraction_rate(self):
        return self._contraction_rate

    @property
    def seed(self):
        return self._seed

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

    @property
    def inner_diameter(self):
        return self._inner_diameter

    @property
    def outer_diameter(self):
        return self._outer_diameter

    @property
    def initial_outer_diameter(self):
        return self._initial_outer_diameter

    @property
    def inner_packing_fraction(self):
        return (4/3 * pi * (self.inner_diameter/2)**3 * self.n_spheres /
                self.domain_volume)

    @property
    def outer_packing_fraction(self):
        return (4/3 * pi * (self.outer_diameter/2)**3 * self.n_spheres /
                self.domain_volume)

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

    @n_spheres.setter
    def n_spheres(self, n_spheres):
        if n_spheres < 0:
            raise ValueError('Unable to set "n_spheres" to {}: number of '
                             'n_spheres must be positive.'.format(n_spheres))
        packing_fraction = self.sphere_volume * n_spheres / self.domain_volume
        if packing_fraction < 0:
            raise ValueError('Unable to set packing fraction to {}: packing '
                             'fraction must be '
                             'positive.'.format(packing_fraction))
        if packing_fraction >= 0.64:
            raise ValueError('Packing fraction of {} is greater than the '
                             'packing fraction limit for close random '
                             'packing (0.64)'.format(packing_fraction))
        self._n_spheres = int(n_spheres)
        self._packing_fraction = packing_fraction

    @packing_fraction.setter
    def packing_fraction(self, packing_fraction):
        if packing_fraction < 0:
            raise ValueError('Unable to set packing fraction to {}: packing '
                             'fraction must be ' \
                             'positive.'.format(packing_fraction))
        if packing_fraction >= 0.64:
            raise ValueError('Packing fraction of {} is greater than the '
                             'packing fraction limit for close random '
                             'packing (0.64)'.format(packing_fraction))
        n_spheres = packing_fraction * self.domain_volume // self.sphere_volume
        if n_spheres < 0:
            raise ValueError('Unable to set "n_spheres" to {}: number of '
                             'n_spheres must be positive.'.format(n_spheres))
        self.n_spheres = n_spheres

    @contraction_rate.setter
    def contraction_rate(self, contraction_rate):
        if contraction_rate < 0:
            raise ValueError('Unable to set "contraction_rate" to {}: '
                             'contraction rate must be '
                             'positive.'.format(contraction_rate))
        self._contraction_rate = contraction_rate

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @inner_diameter.setter
    def inner_diameter(self, inner_diameter):
        self._inner_diameter = inner_diameter

    @outer_diameter.setter
    def outer_diameter(self, outer_diameter):
        self._outer_diameter = outer_diameter

    @initial_outer_diameter.setter
    def initial_outer_diameter(self, initial_outer_diameter):
        self._initial_outer_diameter = initial_outer_diameter


    def _random_points_cube(self):
        """Generate Cartesian coordinates of sphere centers of spheres that are
        contained entirely within cubic domain with uniform probability.

        Returns
        -------
        coordinates : numpy.ndarray
            Cartesian coordinates of sphere centers

        """

        return np.random.uniform(self.radius, self.domain_length - self.radius,
                                 (self.n_spheres, 3))


    def _random_points_cylinder(self):
        """Generate Cartesian coordinates of sphere centers of spheres that are
        contained entirely within cylindrical domain with uniform probability.
        See http://mathworld.wolfram.com/DiskPointPicking.html for generating
        random points on a disk

        Returns
        -------
        coordinates : list
            Cartesian coordinates of sphere center

        """

        t = np.random.uniform(0, 2*pi, self.n_spheres)
        r = np.random.uniform(0, (self.domain_radius - self.radius)**2,
                                  self.n_spheres)**.5
    
        return np.dstack((r*np.cos(t), r*np.sin(t),
                         np.random.uniform(self.radius, self.domain_length -
                         self.radius, self.n_spheres)))[0]


    def _random_points_sphere(self):
        """Generate Cartesian coordinates of sphere centers of spheres that are
        contained entirely within spherical domain with uniform probability.

        Returns
        -------
        coordinates : list
            Cartesian coordinates of sphere center

        """

        x = np.random.normal(0, 1, (self.n_spheres, 3))
        r = np.random.uniform(0, (self.domain_radius - self.radius)**3,
                              self.n_spheres)**(1/3)

        return x * (r/np.sum(x**2, axis=1)**.5)[:, np.newaxis]


    def _create_rod_list(self):
        """Generate sorted list of rods (distances between sphere centers). A
        rod between spheres p and q is only included in the list if the
        distance between p and q could not be changed by the elimination of a
        greater overlap, i.e. q has no nearer neighbors than p.

        """

        # Create KD tree for quick nearest neighbor search
        tree = scipy.spatial.cKDTree(self.spheres)

        # Find distance to nearest neighbor and index of nearest neighbor for
        # all spheres (take the index of and distance to the second nearest
        # neighbor since the nearest neighbor is self)
        d, n = tree.query(self.spheres, k=2)
        d = d[:,1]
        n = n[:,1]

        # Create array of shape [n_spheres, 3] where each row is sphere index,
        # index of nearest neighbor, and distance to nearest neighbor
        a = np.dstack(([i for i in range(len(n))], n, d))[0]

        # Create array of shape [n_spheres, 3] where each row is the index of
        # a nearest neighbor (second column of 'a'), the index of the sphere it
        # is the nearest neighbor of, and the distance between them
        b = a[a[:,1].argsort()]
        b[:,[0, 1]] = b[:,[1, 0]]

        # Find the intersection between p and q: a list of spheres who are each
        # other's nearest neighbors and the distance between them
        self.rods = [x for x in {tuple(x) for x in a} & {tuple(x) for x in b}]

        # Remove duplicate rods and sort by distance
        self.rods = sorted(
            map(list, set([(x[2], int(min(x[0:2])), int(max(x[0:2])))
            for x in self.rods if x[2] <= 2*self.radius])))

        # Inner diameter is set initially to the shortest center-to-center
	# distance between two spheres
        if self.rods:
            self.inner_diameter = self.rods[0][0]


    def _reduce_outer_diameter(self):
	"""Reduce the outer diameter so that at the (i+1)-st iteration it is:

            d_out^(i+1) = d_out^(i) - (1/2)^(j) * d_out0 * k / n,

        where k is the contraction rate, n is the number of spheres, and

            j = floor(-log10(pf_out - pf_in)).

        """

	j = floor(-log10(self.outer_packing_fraction -
                         self.inner_packing_fraction))
	self.outer_diameter = (self.outer_diameter - 0.5**j *
			       self.initial_outer_diameter *
			       self.contraction_rate / self.n_spheres)


    def _repel_spheres_cube(self, i, j, d):
	"""Move spheres p and q apart according to the following transformation
        (accounting for reflective boundary conditions on domain):

            r_i^(n+1) = r_i^(n) + 1/2(d_out^(n+1) - d^(n))
            r_j^(n+1) = r_j^(n) - 1/2(d_out^(n+1) - d^(n))

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array
        d : float
            distance between centers of spheres i and j

        """

	# Moving each sphere distance 'r' away from the other along the line
	# joining the sphere centers will ensure their final distance is equal
	# to the outer diameter
        r = (self.outer_diameter - d)/2;

        v = (self.spheres[i] - self.spheres[j])/d
        self.spheres[i] = self.spheres[i] + r*v
        self.spheres[j] = self.spheres[j] - r*v

        a = self.radius
        b = self.domain_length - a

        # Apply reflective boundary conditions
        for k in range(3):
            if self.spheres[i][k] < a:
                self.spheres[i][k] = a
            elif self.spheres[i][k] > b:
                self.spheres[i][k] = b
            if self.spheres[j][k] < a:
                self.spheres[j][k] = a
            elif self.spheres[j][k] > b:
                self.spheres[j][k] = b


    def _repel_spheres_cylinder(self, i, j, d):
	"""Move spheres p and q apart according to the following transformation
        (accounting for reflective boundary conditions on domain):

            r_i^(n+1) = r_i^(n) + 1/2(d_out^(n+1) - d^(n))
            r_j^(n+1) = r_j^(n) - 1/2(d_out^(n+1) - d^(n))

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array
        d : float
            distance between centers of spheres i and j

        """

	# Moving each sphere distance 'r' away from the other along the line
	# joining the sphere centers will ensure their final distance is equal
	# to the outer diameter
        r = (self.outer_diameter - d)/2;

        v = (self.spheres[i] - self.spheres[j])/d
        self.spheres[i] = self.spheres[i] + r*v
        self.spheres[j] = self.spheres[j] - r*v

        a = (self.radius-self.domain_radius,
             self.radius-self.domain_radius, self.radius)
        b = (self.domain_radius-self.radius,
             self.domain_radius-self.radius, self.domain_length-self.radius)

        # Apply reflective boundary conditions
        for k in range(3):
            if self.spheres[i][k] < a[k]:
                self.spheres[i][k] = a[k]
            elif self.spheres[i][k] > b[k]:
                self.spheres[i][k] = b[k]
            if self.spheres[j][k] < a[k]:
                self.spheres[j][k] = a[k]
            elif self.spheres[j][k] > b[k]:
                self.spheres[j][k] = b[k]


    def _repel_spheres_sphere(self, i, j, d):
	"""Move spheres p and q apart according to the following transformation
        (accounting for reflective boundary conditions on domain):

            r_i^(n+1) = r_i^(n) + 1/2(d_out^(n+1) - d^(n))
            r_j^(n+1) = r_j^(n) - 1/2(d_out^(n+1) - d^(n))

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array
        d : float
            distance between centers of spheres i and j

        """

	# Moving each sphere distance 'r' away from the other along the line
	# joining the sphere centers will ensure their final distance is equal
	# to the outer diameter
        r = (self.outer_diameter - d)/2;

        v = (self.spheres[i] - self.spheres[j])/d
        self.spheres[i] = self.spheres[i] + r*v
        self.spheres[j] = self.spheres[j] - r*v

        a = self.radius - self.domain_radius
        b = self.domain_radius - self.radius

        # Apply reflective boundary conditions
        for k in range(3):
            if self.spheres[i][k] < a:
                self.spheres[i][k] = a
            elif self.spheres[i][k] > b:
                self.spheres[i][k] = b
            if self.spheres[j][k] < a:
                self.spheres[j][k] = a
            elif self.spheres[j][k] > b:
                self.spheres[j][k] = b


    def _nearest(self, i):
        """Find index of nearest neighbor of sphere

        Parameters
        ----------
        i : int
            Index in spheres array of sphere to find nearest neighbor of

        Returns
        -------
        j, d : int, double
            Index in spheres array of nearest neighbor of i; distance between i
            and nearest neighbor

        """

        # Need the second nearest neighbor of i since the nearest neighbor
        # will be itself. Using argpartition, the k-th nearest neighbor is
        # placed at index k.
        dists = cdist([self.spheres[i]], self.spheres)[0]
        j = np.argpartition(dists, 1)[1]
        return j, dists[j]


    def _update_rod_list(self, i, j):
        """Update the rod list with the new nearest neighbors of spheres i and
           j since their overlap was eliminated

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array

        """

        k, d_ik = self._nearest(i)
        l, d_jl = self._nearest(j)

        # If the nearest neighbor k of sphere i has no nearer neighbors, remove
        # the rod containing k from the rod list and add rod k-i keeping rod
        # list sorted
        if self._nearest(k)[0] == i:
            self.rods = [x for x in self.rods if not k in x[1:3]]
            bisect.insort_left(self.rods, [d_ik, min(i,k), max(i,k)])
        if self._nearest(l)[0] == j:
            self.rods = [x for x in self.rods if not l in x[1:3]]
            bisect.insort_left(self.rods, [d_jl, min(j,l), max(j,l)])

        # Set inner diameter to the shortest distance between two sphere
        # centers
        if self.rods:
            self.inner_diameter = self.rods[0][0]


    def pack(self):
        """Randomly place all spheres in domain such that they are not
        overlapping

        Returns
        -------
        spheres : list
            Cartesian coordinates of all spheres in the domain.

        """

        np.random.seed(self.seed)

        diameter = 2*self.radius
        sqdiameter = diameter**2

        # Set domain dependent functions
        if self.geometry is 'cube':
            random_points = self._random_points_cube
            repel_spheres = self._repel_spheres_cube
        elif self.geometry is 'cylinder':
            random_points = self._random_points_cylinder
            repel_spheres = self._repel_spheres_cylinder
        elif self.geometry is 'sphere':
            random_points = self._random_points_sphere
            repel_spheres = self._repel_spheres_sphere

        # TODO: Whether to use np random or built in. Generate non-overlapping
        # spheres for some initial inner radius threshold (using random
        # sequential pack)
        # Randomly choose position of sphere centers within the domain
        self.spheres = random_points()

        while True:

            # Create a sorted list of rods. A rod is the distance between two
            # overlapping spheres. A rod between spheres p and q is only placed on
            # the list if q has no closer neighbors than p.
            self._create_rod_list()

            if self.inner_diameter >= diameter:
                break

            while True:

                # Get indices of the two closest spheres and the distance between
                # their centers
                d, i, j = self.rods.pop(0)

                self._reduce_outer_diameter()

                # Move spheres the two closest spheres apart so they are separated
                # by one diameter
                repel_spheres(i, j, d)

                # Update rod list with new nearest neighbors
                self._update_rod_list(i, j)

                if self.inner_diameter >= diameter or not self.rods:
                    break

        return self.spheres
