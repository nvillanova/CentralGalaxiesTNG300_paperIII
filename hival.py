# Libraries
import auxiliary_dictionaries
from parameter_space import ParameterSpace
from utils import vectorize_get_random_particle, exclude_nans

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay
from scipy.stats import pearsonr

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class HiVAl(ParameterSpace):

    def __init__(self, target_props, nbin=None):
        """
        A HiVAl object is characterized by a set of features/dimensions.
        For example, ['smass', 'color', 'sSFR', 'radius'].
        This is a space with 4 dimensions. This is a 4D parameter/event space which
        will be identified as 'smass_color_sSFR_radius' (name_of_event_space attribute).
        A folder with this name will be created: voronoi_targets/smass_color_sSFR_radius
        (voronoi_folder_props attribute).

        The HiVAl code will categorize this parameter space into domains/cells/classes.
        The number of domains (nbin attribute) depends on the number of iterations.
        (The choice of the number of domains can be made, for example, by balancing
        between occupancy and resolution. The code provides outputs to analyse
        these statistics. =])

        One could choose to consider only a reduced set of features, e.g.,
        ['smass', 'color']. This will define a new event space: 'smass_color'.
        A new folder will be created: voronoi_targets/smass_color/.
        All the files related to this event space will be stored in this folder.

        Given an event space, each iteration of HiVAl will
        create a new folder identified by the number of domains resulting from the
        corresponding iteration, e.g.: voronoi_targets/smass_color/599_classes/
        (voronoi_folder attribute).

        The methods in the HiVAl class are the following.
        - sample_continuous_values
        - assign_HiVAl_domain
        To load the files generated during the iterations:
        - num_classes_list
        - particles
        - positions
        - dispersions
        - volumes
        - occupancy
        - pcc
        - domains (of each instance used to fit HiVAl; ytrain.csv file)
        - target_norm (the data used to fit HiVAl - normalized)
        - stats (to scale and invert scaling as done in the fitting data)
        - get_particles_list (organizes the particles occupying each domain as a list of lists)

        :param target_props: names of the input features (dimensions of the parameter space);
        :param nbin: number of domains/cells/classes;

        :type target_props: list of strings;
        :type nbin: int;
        """

        super().__init__(target_props)

        self.target_props = target_props
        self.nbin = nbin
        # Directory for the parameter space
        self.voronoi_folder_props = f'voronoi_targets/{self.name_of_event_space}/'
        # Directory for a HiVAl iteration (identified by nbin)
        self.voronoi_folder = self.voronoi_folder_props + f'{self.nbin}_classes/'

        if not os.path.exists(self.voronoi_folder_props):
            print('Parameter space does not exist.')
        if not os.path.exists(self.voronoi_folder):
            print('Iteration folder does not exist.')

    def num_classes_list(self):
        return pd.read_csv(self.voronoi_folder_props +
                           'num_classes_list.csv').to_numpy()[:, 0]

    def particles(self):
        return pd.read_csv(self.voronoi_folder + 'particles.csv').to_numpy()

    def positions(self):
        return pd.read_csv(self.voronoi_folder + 'positions.csv').to_numpy()

    def dispersions(self):
        return pd.read_csv(self.voronoi_folder + 'dispersions.csv').to_numpy()

    def volumes(self):
        return pd.read_csv(self.voronoi_folder + 'volumes.csv').to_numpy()[:, 0]

    def occupancy(self):
        return pd.read_csv(self.voronoi_folder + 'occupancy.csv').to_numpy()[:, 0]

    def pcc(self):
        return pd.read_csv(self.voronoi_folder + 'pcc.csv').to_numpy()

    def domains(self):
        """
        :return: Domains/classes of each instance from 'targets_norm.csv'.
        """
        return pd.read_csv(self.voronoi_folder + 'ytrain.csv').to_numpy()[:, 0]

    def target_norm(self):
        """
        :return: Load normalized data used to fit HiVAl.
        """
        return pd.read_csv(self.voronoi_folder_props + 'targets_norm.csv')

    def bin_edges(self):
        """
        For uni-variate distributions, the sampling strategy requires the edges of the
        bins to sample continuous values from a Uniform distribution defined within the
        range of the bin edges.
        :return: bin_edges file
        """
        return pd.read_csv(self.voronoi_folder + 'bin_edges.csv')

    def stats(self):
        """
        :return: minimum and maximum values from the sample used to fit HiVAl
        """
        return pd.read_csv(self.voronoi_folder_props + 'stats.csv')

    def scale(self, sample):
        """
        :param sample: data set
        :return: normalized data set
        """
        stats = self.stats()
        stats_min = stats[self.target_props].to_numpy()[0]
        stats_max = stats[self.target_props].to_numpy()[1]

        sample_scaled = (sample - stats_min) / (stats_max - stats_min)
        return sample_scaled

    def invert_scale(self, sample_scaled):
        """
        :param sample_scaled: normalized data set
        :return: sample without scaling
        """
        stats = self.stats()
        stats_min = stats[self.target_props].to_numpy()[0]
        stats_max = stats[self.target_props].to_numpy()[1]

        sample = sample_scaled * (stats_max - stats_min) + stats_min
        return sample

    def get_particles_list(self):
        """
        Returns a list of lists: the particles contained in each domain.
        Each domain is a list with the indices of the particles there contained.
        """
        particles = self.particles()
        particles_list = []

        for ind_class in range(self.nbin):
            particles_list.append(exclude_nans(particles[ind_class]).astype(int))

        return particles_list

    def sample_continuous_values(self, predicted_classes, sample_size=1):
        """
        Sample continuous value from a domain. This is done by defining a Gaussian
        distribution with mean value corresponding to a particle randomly selected
        from the 'particle_list' of the corresponding domain, and standard deviation
        corresponding to the 'dispersion' of the corresponding class.

        :param predicted_classes: list of domains to sample continuous values from.
        :param sample_size: number of continuous values to sample from each domain.
        """
        particles_list = self.get_particles_list()
        target_norm = self.target_norm().to_numpy()
        dispersions = self.dispersions()

        # The mean value of the Gaussian is a particle randomly chosen from the domain
        mu_particles = vectorize_get_random_particle(np.array(particles_list,
                                                              dtype=object)[predicted_classes]).astype(int)
        mu_vec = target_norm[mu_particles]

        # The std of the Gaussian is the dispersion of the domain
        std_vec = dispersions[predicted_classes]

        # Build Gaussian
        dist = tfd.Normal(mu_vec, std_vec)

        # Sample from Gaussian
        sample = dist.sample(sample_size).numpy()

        # We have been using the target_norm file, therefore these are scaled values
        # convert scaled values to real values
        sample = self.invert_scale(sample)

        return sample

    def assign_HiVAl_cell(self, dataset):
        """
        Assign new particle to a voronoi class. Allocate an instance to a domain.
        """
        # Get dataset instance to be assigned a class
        # the classes are defined based on "target_properties"
        test_halo = dataset[self.target_props].to_numpy()
        # scale the values of the instances to properly fit Voronoi
        test_halo_scaled = self.scale(test_halo)  # , [i for i in range(len(self.target_props))])

        positions = self.positions()

        # Find the cell whose center position is the closest to the coordinates of the instance
        # (smaller difference between vectors)
        nearest_cell = []
        for i in range(len(test_halo_scaled)):
            nearest_cell.append(np.argmin(np.sqrt(np.sum((positions - test_halo_scaled[i]) ** 2, axis=1))))
        nearest_cell = np.array(nearest_cell)

        # Build the particles_list: length = # cells, collection of particles belonging to each cell
        particles_list_new_data = []
        for i in range(self.nbin):
            particles_list_new_data.append(np.where(nearest_cell == i))

        return particles_list_new_data



# Functions to run HiVAl
# - voronoi_volumes
# - voronoi_areas
# - volume_joint
# - intersect2D
# - pairings


def voronoi_volumes(vor):
    # Computes volumes of all Voronoi cells; returns as array (N,2)
    v = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            v[i] = np.inf
        else:
            v[i] = ConvexHull(vor.vertices[indices]).volume
    return v


def voronoi_areas(vor):
    # Computes areas of all Voronoi cells; returns as array (N,2)
    a = np.zeros(vor.npoints)
    for i, reg_num in enumerate(vor.point_region):
        indices = vor.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            a[i] = np.inf
        else:
            a[i] = ConvexHull(vor.vertices[indices]).area
    return a


def volume_joint(points):
    # If shared face is a line, use the length of that line
    npoints, dim = points.shape[0], points.shape[1]
    if dim == 2:
        len_vol = np.sum(np.sqrt(np.sum((points-np.roll(points, 1, axis=0))**2, axis=1)))
    # If shared face is 2D, use the area. Notice that a triangle in 3D does NOT define a convex hull!
    elif dim == 3:
        if npoints == 3:
            len_vol = np.sqrt(np.sum(np.cross(points[0]-points[1], points[2]-points[1])/2)**2)
        else:
            len_vol = ConvexHull(points, qhull_options='QJ').area
    # Otherwise, use generalized volume. Again, in D dimensions the number of points of a
    # convex hull must be D + 1. If there are less point than that, define volume = 0.
    else:
        if npoints == dim:
            len_vol = 0.0
        else:
            len_vol = ConvexHull(points, qhull_options='QJ').volume
    return len_vol


def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def pairings(positions, list_particles_in, verbose=False):
    """
    This function takes in the coordinates (positions) and memberships (list_particles_in)
    of a given set of domains. It also needs a list determining whether the Voronoi
    domains ("positions") have finite or infinite volumes.
    It then "pairs up" neighboring domains, starting with the ones of smallest volume.
    :param positions: coordinates of the center of the domains in the parameter space
    :param list_particles_in: particles (instances) that occupy each domain
    :param verbose: Boolean; choose whether to output comments or not
    :return: positions of updated domains, list of particles belonging to updated domains,
    volumes of updated domains
    """

    ndim = np.shape(positions)[1]

    vor = Voronoi(positions)
    volumes = voronoi_volumes(vor)

    # Reorder from smallest to larges volume (i.e., ends with volume=inf cells @ edge)
    ord_vol = np.argsort(volumes)
    positions = positions[ord_vol]
    volumes = volumes[ord_vol]
    list_particles_in = [list_particles_in[i] for i in ord_vol]
    if verbose:
        print("Positions of the cells on this iteration:")
        print(positions)

    # Redo Voronoi
    vor = Voronoi(positions)

    # Point region is the index of the domain -- used to find the vertices
    point_regions = np.asarray(vor.point_region)
    regions = vor.regions    # These are the indices that tell which vertices belong to which points
    vertices = vor.vertices

    # This finds the neighbouring cells for all given Voronoi cells
    ind_neigh, neighbours = Delaunay(positions).vertex_neighbor_vertices

    # Initialize lists for pairings
    list_finite_vol = np.logical_not(np.isinf(volumes))

    list_not_paired = np.arange(0, len(positions))
    list_paired = -1*np.ones(len(positions))
    # list_paired_with = -1*np.ones(len(positions))
    list_single = -1*np.ones(len(positions))

    list_unique_pair_1 = -1*np.ones(len(positions))
    list_unique_pair_2 = -1*np.ones(len(positions))

    # Mark infinite cells as single; will not be paired
    list_not_paired[np.logical_not(list_finite_vol)] = -1

    num_unpaired = 0
    count = 0

    for i, point_region in enumerate(point_regions):

        if np.mod(i, 9999) == 0:
            Ncount = 10000*count
            count = count + 1
            if count > 1:
                print("Paired ", Ncount, "particles...")

        # Position of i-th particle
        pos_i = positions[i]

        if list_finite_vol[i]:
            if verbose:
                print()
                print("i=", i)
                print("List paired:", list_paired)
                print("List not paired:", list_not_paired)

            # Check if cell not already paired with other cell
            if not np.any(list_paired == i):
                # Find vertices for this domain
                i_vr = np.asarray(regions[point_region])
                i_vt = np.asarray(vertices[i_vr])
                if verbose:
                    print("i vertices:", i_vt)
                i_neigh = neighbours[ind_neigh[i]:ind_neigh[i+1]]
                # Positions of neighbors
                pos_neigh = positions[i_neigh]
                # Compute distances to all members; pick only 3 nearest neighbors
                n_neigh = len(i_neigh)
                pos_i_array = np.tile(pos_i, (n_neigh, 1))
                dist_array = np.sqrt(np.sum((pos_i_array - pos_neigh)**2, axis=1))
                #
                # Find 3 nearest neighbors
                # THIS IS ARBITRARY...!
                nn_3 = i_neigh[np.argsort(dist_array)[:3]]
                # Get neighbors which are not already taken
                if verbose:
                    print("  Nearest neighbors:", i_neigh[np.argsort(dist_array)])
                intersect = np.intersect1d(list_not_paired, i_neigh)
                intersect = np.intersect1d(list_not_paired, nn_3)
                if verbose:
                    print("  Intersect =", intersect)
                # Sometimes the neighbors are all taken, so intersect = []
                if len(intersect) > 0:
                    # For each neighbor, get the one with the highest common side (2D),
                    # face (3D) or volume (4D or more)
                    len_joint = 0.0
                    for j in intersect:
                        if verbose:
                            print("  j=", j)
                        j_pr = np.asarray(point_regions[j])
                        j_vr = np.asarray(regions[j_pr])
                        j_vt = np.asarray(vertices[j_vr])
                        if verbose:
                            print("  j vertices:", j_vt)

                        # Find set of common vertices between j_vt and i_vt
                        j_joint = intersect2D(i_vt, j_vt)
                        # If the two cells are neighbors but only by a vertex, volume is zero.
                        # Otherwise, compute the volume -- but always take dimension of common
                        if len(np.shape(j_joint)) > 1:
                            j_dim = np.shape(j_joint)[1]
                            if j_dim == ndim:
                                if verbose:
                                    print("  j_joint=", j_joint)
                                len_j = volume_joint(j_joint)
                                if verbose:
                                    print("  v_joint=", len_j)
                                # Now, pick the highest one
                                if len_j > len_joint:
                                    len_joint = len_j
                                    best_neigh = j
                                    if verbose:
                                        print("  Best j=", best_neigh, " v_b=", len_joint)
                                        print()
                    # If none of 3 nearest neighboring cells satisfy the requirement above,
                    # then skip that cell
                    if len_joint == 0:
                        if verbose:
                            print("   --> len_joint = 0 for cell i=", i)
                        list_single[i] = i
                        num_unpaired = num_unpaired + 1
                    else:
                        list_paired[best_neigh] = best_neigh
                        list_paired[i] = i
                        list_not_paired[i] = -1
                        list_not_paired[best_neigh] = -1
                        # list_paired_with[i] = best_neigh
                        # list_paired_with[best_neigh] = i
                        list_unique_pair_1[i] = i
                        list_unique_pair_2[i] = best_neigh
                else:
                    # If all 3 nearest neighboring cells are either inf. vol. or have been taken,
                    # then skip that cell
                    if verbose:
                        print("   --> All neighbors already taken for cell i=", i)
                        print()
                    # CHANGE JULY 3: finite vol. cells "not paired" should be left available for pairing
                    # list_single[i]= i
                    # list_paired[i]= -1
                    # list_not_paired[i]= -1
                    # num_unpaired = num_unpaired + 1
        else:  # Cell has infinite volume; leave as single
            if verbose:
                print("  Cell i=", i, "has infinite volume")
                print()
            list_single[i] = i
            num_unpaired = num_unpaired + 1

    # CHANGE JULY 4:
    # list_len = len(list_unique_pair_1[list_unique_pair_1>-1]) + len(list_single[list_single>-1])
    # list_len = len(list_unique_pair_1[list_unique_pair_1 > -1]) +
    #            len(list_single[list_single > -1]) +
    #            len(list_not_paired[list_not_paired > -1])
    l1 = len(list_unique_pair_1[list_unique_pair_1 > -1])
    l2 = len(list_not_paired[list_not_paired > -1])
    l3 = len(list_single[list_single > -1])
    list_len = l1 + l2 + l3
    print("Number of 'paired' domains:", l1)
    print("Number of 'not paired' domains:", l2)
    print("Number of 'single' domains:", l3)
    print("Total number of particles at the end (paired + singles + not paired) = ", list_len)

    # Initialize list of objects in each site (cell)

    # Prepare the output list, with the domains and their compositions
    list_particles_out = [[] for _ in range(list_len)]

    lp = np.int0(np.array([list_unique_pair_1[list_unique_pair_1 > -1],
                           list_unique_pair_2[list_unique_pair_1 > -1]]).T)
    lu = np.int0(list_not_paired[list_not_paired > -1])
    ls = np.int0(list_single[list_single > -1])
    if verbose:
        print("List unique pairs (1):", lp)
        print("List not paired:", lu)
        print("List single:", ls)
    # CHANGE JULY 4:
    # lp_len = len(lp)
    # ls_len = len(ls)
    lp_len = l1
    lu_len = l2
    ls_len = l3

    # Add pairs to list of "new particles" (here I assume the initial, empty, list)
    # for i in range(len(lp)):
    for i in range(l1):
        list_particles_out[i].extend(list_particles_in[lp[i, 0]])
        list_particles_out[i].extend(list_particles_in[lp[i, 1]])

    # CHANGE JULY 4:
    # Add particles in "not paired" domains
    for j in range(l2):
        jlist = l1 + j
        list_particles_out[jlist].extend(list_particles_in[lu[j]])

    # Now single particles
    for j in range(l3):
        jlist = l1 + l2 + j
        list_particles_out[jlist].extend(list_particles_in[ls[j]])

    # Now, the positions of the "new" domains
    # The "center of mass" will be on the lines joining the pair, weighted by the inverse volumes
    pa = np.int0(list_unique_pair_1[list_unique_pair_1 > -1])
    pb = np.int0(list_unique_pair_2[list_unique_pair_2 > -1])
    # weights

    # ndim = 1
    wa = np.power(volumes[pa], -1./ndim)
    wb = np.power(volumes[pb], -1./ndim)

    # print("Unique pairs:", np.array([list_unique_pair_1,list_unique_pair_2]).T)

    cm = ((wa*positions[pa].T + wb*positions[pb].T)/(wa+wb)).T
    # Positions of the single domains
    # print("Index a,b:", np.array([pa,pb]).T )
    # print("Positions:", np.hstack((positions[pa],positions[pb])) )
    # print("Weights:", np.array([wa,wb]).T )
    # print("Pair positions:", np.array(cm) )

    unpaired = positions[list_not_paired > -1]
    singles = positions[list_single > -1]
    positions_out = np.vstack([cm, unpaired, singles])

    return positions_out, list_particles_out, volumes


# NVNR contribution
# - standardize
# - save_voronoi


def standardize(data, hival_object):
    """
    :type data: input data to fit HiVAl; pandas.DataFrame
    :type hival_object: HiVAl object (class)
    """
    target_props = hival_object.target_props
    voronoi_folder_props = hival_object.voronoi_folder_props

    targets = data[target_props]

    print('Generating target_norm.csv and stats.csv files.')

    # Save max and min of properties
    stats = np.vstack((targets.min(axis=0), targets.max(axis=0)))
    stats = pd.DataFrame(stats, columns=[target_props])
    stats.to_csv(voronoi_folder_props + 'stats.csv', index=False)

    # Save scaled object properties
    targets = (targets - targets.min(axis=0)) / (targets.max(axis=0) - targets.min(axis=0))
    # targets = targets * (1 - 0) + 0
    targets.to_csv(voronoi_folder_props + 'targets_norm.csv', index=False)


def save_voronoi(i, num_iter, hival_object, pos_aux,
                 positions, particles_list, volumes):
    """
    Stuff to be saved at each iteration.
    :param i: iteration index
    :param num_iter:
    :param hival_object: HiVAl object (class)
    :param pos_aux:
    :param positions: positions of the center of the domains (HiVAl output)
    :param particles_list: list of particles occupying the domains (HiVAl output)
    :param volumes: volumes of the domains (HiVAl output)
    :return:
    """
    voronoi_folder_props = hival_object.voronoi_folder_props
    target_props = hival_object.target_props
    pairs = hival_object.pairs
    Ndims = hival_object.Ndims

    num_classes = len(positions)
    print('save_voronoi: num_classes = ', num_classes)

    # Crate directory to save the files of the corresponding iteration
    voronoi_folder = voronoi_folder_props + '{}_classes/'.format(num_classes)

    if os.path.exists(voronoi_folder):
        print('ignoring: FILE EXISTS! All the content will be overwritten!')
    # os.mkdir(path=dir_name + '_v')
    else:
        os.mkdir(path=voronoi_folder)

    # Save Voronoi cell positions, particles list and volumes
    pd.DataFrame(positions).to_csv(voronoi_folder + 'positions.csv', index=False)
    pd.DataFrame(particles_list).to_csv(voronoi_folder + 'particles.csv', index=False)
    pd.DataFrame(volumes).to_csv(voronoi_folder + 'volumes.csv', index=False)

    # Sanity checks

    # visualize Voronoi diagram (2D plots)
    if num_classes < 10000:
        for pair_ind in pairs:

            x_ind = pair_ind[0]
            y_ind = pair_ind[1]

            pair_name = target_props[x_ind] + '_' + target_props[y_ind]

            pos_2d = positions[:, pair_ind]

            vor_2d = Voronoi(pos_2d)
            voronoi_plot_2d(vor_2d)
            plt.title(pair_name)
            plt.savefig(voronoi_folder + 'diagram_{}.png'.format(pair_name))
            if i == num_iter - 1:
                plt.show()
            plt.close()

        # individual properties histogram
        bins = np.linspace(-5, 5, 50)

        for prop_ind in range(Ndims):
            pos_1d = positions[:, prop_ind]
            hist = np.histogram(pos_1d, bins=bins, density=True)[0]

            hist_iter0 = np.histogram(pos_aux[0][:, prop_ind], bins=bins, density=True)[0]

            plt.rcParams['figure.figsize'] = [5, 5]
            plt.plot(hist_iter0, label='iter.0', color='k', ls='--')
            plt.plot(hist, label='iter.{}'.format(i))
            plt.title(target_props[prop_ind])
            plt.legend()
            plt.savefig(voronoi_folder + 'hist_{}.png'.format(target_props[prop_ind]))
            if i == num_iter - 1:
                plt.show()
            plt.close()


def save_classes_and_dispersions(hival_object, save=True):

    print('Loading attributes.')
    voronoi_folder = hival_object.voronoi_folder()
    target_norm = hival_object.target_norm()
    particles = hival_object.particles()
    target_props = hival_object.target_props
    num_classes = hival_object.nbin

    ytotal = np.zeros(len(target_norm))

    std_cell = pd.DataFrame(np.zeros((num_classes, len(target_props))),
                            columns=[target_props])

    print('Assigning classes and dispersions.')
    for ind_class in range(len(particles)):
        p_ = particles[ind_class]
        p_ = p_[~np.isnan(p_)]

        # classes
        # to each particle is associated a class (domain which it belongs to)
        ytotal[p_.astype(int)] = ind_class

        # dispersions (in terms of standardized values)
        # use values from the file (target_norm) used to run
        std_cell.iloc[ind_class] = np.std(target_norm[target_props].iloc[p_.astype(int)])

    # Save files
    if save:
        print('Saving ytrain.csv and dispersions.csv')
        pd.DataFrame(ytotal.astype(int)).to_csv(voronoi_folder + 'ytrain.csv', index=False)
        pd.DataFrame(std_cell).to_csv(voronoi_folder + 'dispersions.csv', index=False)


def run_HiVAl(data, target_props, num_iter=30, min_num_classes=5000):

    os.makedirs('voronoi_targets', exist_ok=True)

    if len(target_props) == 1:
        univariate_case(data, target_props, nbin=50)

    else:
        # Create object
        hival_object = HiVAl(target_props)
        voronoi_folder_props = hival_object.voronoi_folder_props
        if not os.path.exists(voronoi_folder_props):
            print('Creating {} file.'.format(voronoi_folder_props))
            os.mkdir(path=voronoi_folder_props)
        else:
            print('Attention: event space exists.')
            # return

        # Standardize: generate targets_norms.csv and stats.csv files
        standardize(data, hival_object)

        # Run HiVAl
        target_norm = hival_object.target_norm().to_numpy()
        Ndims = hival_object.Ndims

        # Initialize positions (center of cells) and particles (that occupy each cell) list
        positions = target_norm
        list_particles_in = [[i] for i in range(len(positions))]

        pos_aux = [positions]  # positions of the center of the cells
        part_aux = [list_particles_in]  # instances that built the cell

        num_classes_list = [len(target_norm)]  # record the number of classes

        for i in range(1, num_iter):

            print('\niteration: ', i)
            start = time.time()

            pos_aux_i, part_aux_i, volumes_i = pairings(pos_aux[i - 1], part_aux[i - 1])
            pos_aux.append(pos_aux_i)
            part_aux.append(part_aux_i)

            # Save
            num_classes_list.append(len(pos_aux_i))
            if len(pos_aux_i) < min_num_classes:
                print('Saving iteration files.')
                # Stop loop when the number of classes stops decreasing
                if num_classes_list[i - 1] == num_classes_list[i]:
                    break
                save_voronoi(i, num_iter, hival_object, pos_aux,
                             pos_aux_i, part_aux_i, volumes_i)

            end = time.time()
            elapsed_time = end - start
            print("Time elapsed:", elapsed_time)

        # Save number of classes of each iteration
        pd.DataFrame(num_classes_list).to_csv(voronoi_folder_props + 'num_classes_list.csv',
                                              index=False)

        # Additional savings (after all iterations)
        print('\nAdditional savings.')
        save = False

        for iter in range(1, len(num_classes_list) - 1):
            if num_classes_list[iter] < min_num_classes:
                save = True

            if save:
                hival_object.nbin = num_classes_list[iter]
                print('\n# domains: ', hival_object.nbin)
                voronoi_folder = hival_object.voronoi_folder
                particles_list = hival_object.get_particles_list()

                # Save classes (ytrain) and dispersions of each cell
                save_classes_and_dispersions(hival_object, save=save)

                # Occupancy = number of particles belonging to a cell
                occ = []
                for j in range(len(particles_list)):
                    occ.append(len(particles_list[j]))
                pd.DataFrame(occ).to_csv(voronoi_folder + 'occupancy.csv', index=False)

                # PCC of perfect class
                true_class = hival_object.domains()
                samp = hival_object.sample_continuous_values(true_class, 1)
                pcc = np.zeros(len(target_props))
                for p in range(len(pcc)):
                    pcc[p] = pearsonr(data[target_props].to_numpy()[:, p], samp[0, :, p])[0]
                pd.DataFrame(pcc[:, np.newaxis].T,
                             columns=target_props).to_csv(voronoi_folder + 'pcc.csv',
                                                          index=False)


def univariate_case(train, target_props, nbin):

    # Create folder
    # create object
    hival_object = HiVAl(target_props, nbin=nbin)
    # event space folder
    voronoi_folder_props = hival_object.voronoi_folder_props
    if not os.path.exists(voronoi_folder_props):
        print('Creating {} file.'.format(voronoi_folder_props))
        os.mkdir(path=voronoi_folder_props)
    else:
        print('Attention: event space exists.')

    # Voronoi_folder
    voronoi_folder = hival_object.voronoi_folder
    if not os.path.exists(voronoi_folder):
        print('Creating {} file.'.format(voronoi_folder))
        os.mkdir(path=voronoi_folder)
    else:
        print('Attention: voronoi folder exists.')

    # save bin_edges
    range_dict = {'smass': (8.75, 13.0),
                  'color': (-0.3, 1.22),
                  'sSFR': (-17, -8.3),
                  'radius': (0.05, 2.2)}

    l, r = range_dict[target_props[0]]
    vec = np.linspace(l, r, nbin)
    bin_edges = np.histogram(train[target_props].to_numpy()[:, 0], bins=vec)[1]

    pd.DataFrame(bin_edges).to_csv('{}/bin_edges.csv'.format(voronoi_folder), index=False)

    ytrain = train[target_props].to_numpy()
    ytrain = np.digitize(ytrain, vec) - 1

    pd.DataFrame(ytrain).to_csv('{}/ytrain.csv'.format(voronoi_folder), index=False)
