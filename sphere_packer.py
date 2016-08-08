from __future__ import division
import numpy as np
import itertools
import scipy.spatial
import random
from random import uniform, gauss
import heapq
from heapq import heappush, heappop
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
    n_spheres : int
        Number of spheres to pack in domain. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    packing_fraction : float
        Final desired packing fraction of spheres. Exactly one of 'spheres' and
        'packing_fraction' should be specified
    contraction_rate : float
        Contraction rate of outer diameter
    lattice_dimension : list
        Number of lattice cells in each dimension
    cell_length : list
        Length of lattice cells in each dimension
    seed : int
        RNG seed
    sphere_volume : float
        Volume of each sphere
    domain_volume : float
        Volume of domain
    diameter : float
        Final diameter of spheres
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
	List of rods arranged in a heap where each element contains the
        distance between the sphere centers and the indices of the two spheres
    rods_map : dict
	Mapping of sphere ids to rods. Each key in the dict is the id of a
        sphere that is in the rod list, and the value is the id of its nearest
	neighbor and the rod that contains them. The dict is used to find rods
	in the priority queue and to mark removed rods so rods can be "removed"
        without breaking the heap structure invariant.
    mesh : dict
        Each key is the index of a lattice cell. The value is a set containing
        the sphere ids of any sphere whose center is within one diameter of
        that cell.
    mesh_map : dict
        Each key in the dict is the id of a sphere. The value is a set
        containing the index of each lattice cell the sphere center is within
        one diameter of.

    """

    def __init__(self, radius, geometry='cylinder', domain_length=None,
                 domain_radius=None, n_spheres=None, packing_fraction=None,
                 contraction_rate=None, lattice_dimension=None, seed=1):

        # Initialize RandomSequentialPacker class attributes
        self._n_spheres = None
        self._packing_fraction = None
        self._radius = None
        self._geometry = None
        self._domain_length = None
        self._domain_radius = None
        self._contraction_rate = None
        self._lattice_dimension = None
        self._seed = None
        self._cell_length = None
        self._diameter = None
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

        if self.geometry is 'cube':
            self.cell_length = [self.domain_length/i for i in
                                self.lattice_dimension]
            self.bc_min = 3*(self.radius,)
            self.bc_max = 3*(self.domain_length-self.radius,)
            self.random_point = self._random_point_cube
            self.cell_list = self._cell_list_cube
            self.cell_index = self._cell_index_cube
        elif self.geometry is 'cylinder':
            self.cell_length = [
                2*self.domain_radius/self.lattice_dimension[0],
                2*self.domain_radius/self.lattice_dimension[1],
                self.domain_length/self.lattice_dimension[2]]
            self.bc_min = (self.radius-self.domain_radius,
                           self.radius-self.domain_radius, self.radius)
	    self.bc_max = (self.domain_radius-self.radius,
			   self.domain_radius-self.radius,
                           self.domain_length-self.radius)
            self.random_point = self._random_point_cylinder
            self.cell_list = self._cell_list_cylinder
            self.cell_index = self._cell_index_cylinder
        elif self.geometry is 'sphere':
            self.cell_length = [2*self.domain_radius/i for i in
                                self.lattice_dimension]
            self.bc_min = 3*(self.radius-self.domain_radius,)
            self.bc_max = 3*(self.domain_radius-self.radius,)
            self.random_point = self._random_point_sphere
            self.cell_list = self._cell_list_sphere
            self.cell_index = self._cell_index_sphere

        if contraction_rate is not None:
            self.contraction_rate = contraction_rate
        else:
            self.contraction_rate = 1/400

        self.seed = seed
        self.diameter = 2*self.radius
        self.initial_outer_diameter = 2 * (self.domain_volume / 
                                          (self.n_spheres * 4/3 * pi))**(1/3)
        self.outer_diameter = self.initial_outer_diameter
        self.initial_packing_fraction = 0.33
        if self.initial_packing_fraction > self.packing_fraction:
            self.initial_packing_fraction = self.packing_fraction
        self.spheres = []
        self.rods = []
        self.rods_map = {}
        self.mesh = defaultdict(set)
        self.mesh_map = defaultdict(set)

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

    @property
    def diameter(self):
        return self._diameter

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

    @diameter.setter
    def diameter(self, diameter):
        self._diameter = diameter

    @inner_diameter.setter
    def inner_diameter(self, inner_diameter):
        self._inner_diameter = inner_diameter

    @outer_diameter.setter
    def outer_diameter(self, outer_diameter):
        self._outer_diameter = outer_diameter

    @initial_outer_diameter.setter
    def initial_outer_diameter(self, initial_outer_diameter):
        self._initial_outer_diameter = initial_outer_diameter


    def _random_point_cube(self):
        """Generate Cartesian coordinates of sphere center of a sphere that is
        contained entirely within cubic domain with uniform probability.

        Returns
        -------
        coordinates : list

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
        coordinates : list
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
        coordinates : list
            Cartesian coordinates of sphere center

        """

        x = (gauss(0, 1), gauss(0, 1), gauss(0, 1))
	r = (uniform(0, (self.domain_radius - self.radius)**3)**(1/3) /
             (x[0]**2 + x[1]**2 + x[2]**2)**.5)
        return [r*i for i in x]


    def _add_rod(self, d, i, j):
        """Add a new rod to the priority queue.

        Parameters
        ----------
        d : float
            distance between centers of spheres i and j
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array

        """

        rod = [d, i, j]
        self.rods_map[i] = j, rod
        self.rods_map[j] = i, rod
        heappush(self.rods, rod)


    def _remove_rod(self, i):
        """Mark the rod containing sphere i as removed.

        Parameters
        ----------
        i : int
            Index of sphere in spheres array

        """

        if i in self.rods_map:
            j, rod = self.rods_map.pop(i)
            del self.rods_map[j]
            rod[1] = None
            rod[2] = None


    def _pop_rod(self):
	"""Remove and return the shortest rod.

        Returns
        -------
        d : float
            distance between centers of spheres i and j
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array


        """

        while self.rods:
            d, i, j = heappop(self.rods)
            if i is not None and j is not None:
                del self.rods_map[i]
                del self.rods_map[j]
                return d, i, j


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
        r = [x for x in {tuple(x) for x in a} & {tuple(x) for x in b}]

        # Remove duplicate rods and sort by distance
        r = map(list, set([(x[2], int(min(x[0:2])), int(max(x[0:2])))
                for x in r]))

        # Clear priority queue and add rods
        del self.rods[:]
        self.rods_map.clear()
        for d, i, j in r:
            self._add_rod(d, i, j)

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


    def _update_mesh(self, i):
	"""Update which lattice cells the sphere is in based on new sphere
        center coordinates.

        Parameters
        ----------
        i : int
            Index of sphere in spheres array

        """

        # Determine which lattice cells the sphere is in and remove the
        # sphere id from those cells
        for idx in self.mesh_map[i]:
            self.mesh[idx].remove(i)
        del self.mesh_map[i]

        # Determine which lattice cells are within one diameter of sphere's
        # center and add this sphere to the list of spheres in those cells
        for idx in self._cell_list_cube(i, self.diameter):
            self.mesh[idx].add(i)
            self.mesh_map[i].add(idx)


    def _apply_boundary_conditions(self, i, j):
        """Apply reflective boundary conditions to spheres i and j

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        j : int
            Index of sphere in spheres array

        """

        for k in range(3):
            if self.spheres[i][k] < self.bc_min[k]:
                self.spheres[i][k] = self.bc_min[k]
            elif self.spheres[i][k] > self.bc_max[k]:
                self.spheres[i][k] = self.bc_max[k]
            if self.spheres[j][k] < self.bc_min[k]:
                self.spheres[j][k] = self.bc_min[k]
            elif self.spheres[j][k] > self.bc_max[k]:
                self.spheres[j][k] = self.bc_max[k]


    def _repel_spheres(self, i, j, d):
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

        # Apply reflective boundary conditions
        self._apply_boundary_conditions(i, j)

        self._update_mesh(i)
        self._update_mesh(j)


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
        idx = list(self.mesh[self.cell_index(self.spheres[i])])
        dists = cdist([self.spheres[i]], self.spheres[idx])[0]
        try:
            j = dists.argpartition(1)[1]
            return idx[j], dists[j]
        except:
            return None, None


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

        # If the nearest neighbor k of sphere i has no nearer neighbors, remove
        # the rod containing k from the rod list and add rod k-i keeping rod
        # list sorted
        k, d_ik = self._nearest(i)
        if k and self._nearest(k)[0] == i:
            self._remove_rod(k)
            self._add_rod(d_ik, i, k)
        l, d_jl = self._nearest(j)
        if l and self._nearest(l)[0] == j:
            self._remove_rod(l)
            self._add_rod(d_jl, j, l)

        # Set inner diameter to the shortest distance between two sphere
        # centers
        if self.rods:
            self.inner_diameter = self.rods[0][0]


    def _cell_index_cube(self, p):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        p : list
            Cartesian coordinates of sphere center

        Returns
        -------
        index : tuple
            indices of lattice cell

        """

        return tuple(int(p[i]/self.cell_length[i]) for i in range(3))


    def _cell_index_cylinder(self, i):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        i : int
            Index of sphere in spheres array

        Returns
        -------
        index : tuple
            indices of lattice cell

        """

        return tuple([int((self.spheres[i][0] + self.domain_radius)/self.cell_length[0]),
                     int((self.spheres[i][1] + self.domain_radius)/self.cell_length[1]),
                     int(self.spheres[i][2]/self.cell_length[2])])


    def _cell_index_sphere(self, i):
        """Calculate the index of the lattice cell the given sphere center
        falls in

        Parameters
        ----------
        i : int
            Index of sphere in spheres array

        Returns
        -------
        index : tuple
            indices of lattice cell

        """

        return tuple(int((self.spheres[i][j] + self.domain_radius)/self.cell_length[j])
                     for j in range(3))


    def _cell_list_cube(self, i, d):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        d : float
	    Find all lattice cells that are within a radius of length 'd' from
            the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

        r = [[a/self.cell_length[j] for a in
             [self.spheres[i][j]-d, self.spheres[i][j], self.spheres[i][j]+d]
             if a > 0 and a < self.domain_length] for j in range(3)]

        return list(itertools.product(*({int(j) for j in k} for k in r)))


    def _cell_list_cylinder(self, i, d):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        d : float
	    Find all lattice cells that are within a radius of length 'd' from
            the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

	x,y = [[(a + self.domain_radius)/self.cell_length[j] for a in
	       [self.spheres[i][j]-d, self.spheres[i][j], self.spheres[i][j]+d]
               if a > -self.domain_radius and a < self.domain_radius]
               for j in range(2)]

	z = [a/self.cell_length[2] for a in
             [self.spheres[i][2]-d, self.spheres[i][2], self.spheres[i][2]+d]
             if a > 0 and a < self.domain_length]

        return list(itertools.product(*({int(j) for j in k} for k in (x,y,z))))


    def _cell_list_sphere(self, i, d):
        """Return the indices of all cells within the given distance of the
        point.

        Parameters
        ----------
        i : int
            Index of sphere in spheres array
        d : float
	    Find all lattice cells that are within a radius of length 'd' from
            the sphere center

        Returns
        -------
        indices : list of tuples
            indices of lattice cells

        """

        r = [[(a + self.domain_radius)/self.cell_length[j] for a in
	     [self.spheres[i][j]-d, self.spheres[i][j], self.spheres[i][j]+d]
             if a > -self.domain_radius and a < self.domain_radius]
             for j in range(3)]

        return list(itertools.product(*({int(j) for j in k} for k in r)))


    def _set_initial_parameters(self):
        """Set parameters for initial random sequential packing of spheres.

        """

	self.radius = (3 * self.initial_packing_fraction * self.domain_volume /
                       (4 * pi * self.n_spheres))**(1/3)

        self.diameter = 2*self.radius

        n = int(self.domain_length/(4*self.radius))
        m = int(self.domain_radius/(2*self.radius))
        if self.geometry is 'cube':
            self.lattice_dimension = [n, n, n]
            self.cell_length = [self.domain_length/i for i in
                                self.lattice_dimension]
        elif self.geometry is 'cylinder':
            self.lattice_dimension = [m, m, n]
            self.cell_length = [
                2*self.domain_radius/self.lattice_dimension[0],
                2*self.domain_radius/self.lattice_dimension[1],
                self.domain_length/self.lattice_dimension[2]]
        elif self.geometry is 'sphere':
            self.lattice_dimension = [m, m, m]
            self.cell_length = [2*self.domain_radius/i for i in
                                self.lattice_dimension]


    def _initialize_spheres(self):
	"""Initial random sequential packing of spheres. This is done to speed
	up the algorithm since for small packing fractions random sequential
        packing is fast. Rather than choosing random coordinates for the sphere
        centers, the spheres are placed randomly to be at least some inital
        distance apart.

        """

        # Save the parameters we want to use for the close random pack
        radius = self.radius
        diameter = self.diameter
        lattice_dimension = self.lattice_dimension
        cell_length = self.cell_length

        self._set_initial_parameters()

	# Need the radius to be the final radius so that random points are
        # selected far enough from boundary
        self.radius = radius

        sqdiameter = self.diameter**2

        for i in range(self.n_spheres):

            # Randomly choose position of sphere center within the domain and
            # continue sampling new center coordinates as long as there are any
            # overlaps
            while True:
               p = self.random_point()
               idx = self.cell_index(p)
               if any((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2 < sqdiameter
                      for q in [self.spheres[j] for j in self.mesh[idx]]):
                   continue
               else:
                   break
            self.spheres.append(p)

            # Determine which lattice cells are within one diameter of sphere's
            # center and add this sphere to the list of spheres in those cells
            for idx in self.cell_list(i, self.diameter):
                self.mesh[idx].add(i)

        self.spheres = np.array(self.spheres)

        # Reset parameters
        self.diameter = diameter
        self.lattice_dimension = lattice_dimension
        self.cell_length = cell_length

        # Determine which lattice cells are within one diameter of sphere's
        # center and add this sphere to the list of spheres in those cells
        self.mesh = defaultdict(set)
        for i in range(self.n_spheres):
            for idx in self.cell_list(i, self.diameter):
                self.mesh[idx].add(i)
                self.mesh_map[i].add(idx)


    def pack(self):
        """Randomly place all spheres in domain such that they are not
        overlapping

        Returns
        -------
        spheres : list
            Cartesian coordinates of all spheres in the domain.

        """

        random.seed(self.seed)

        # Generate non-overlapping spheres for an initial inner radius using
        # random sequential pack)
        self._initialize_spheres()

        if self.initial_packing_fraction == self.packing_fraction:
            return self.spheres

        while True:

            # Create a sorted list of rods. A rod is the distance between two
            # overlapping spheres. A rod between spheres p and q is only placed on
            # the list if q has no closer neighbors than p.
            self._create_rod_list()

            if self.inner_diameter >= self.diameter:
                break

            while True:

                # Get indices of the two closest spheres and the distance between
                # their centers
                d, i, j = self._pop_rod()

                self._reduce_outer_diameter()

                # Move spheres the two closest spheres apart so they are separated
                # by one diameter
                self._repel_spheres(i, j, d)

                # Update rod list with new nearest neighbors
                self._update_rod_list(i, j)

                if self.inner_diameter >= self.diameter or not self.rods:
                    break

        print "Inner diameter: {}".format(self.inner_diameter)
        print "Outer diameter: {}".format(self.outer_diameter)
        print "Inner packing fraction: {}".format(self.inner_packing_fraction)
        print "Outer packing fraction: {}".format(self.outer_packing_fraction)

        return self.spheres
