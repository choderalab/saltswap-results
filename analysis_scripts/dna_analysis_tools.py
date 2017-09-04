import numpy as np
from netCDF4 import Dataset
import mdtraj

def load_simulation(directory, repeat_number):
    """
    Read in saltswap simulation data. It is expected simulation data follows the same naming convention.

    Parameters
    ----------
    directory: str
        The location of the simulation data
    repeat_number: int
        the simulation number.
    """
    # Load files:
    netcdf_filename = directory + '/out{0}.nc'.format(repeat_number)
    pdb_filename = directory + '/out{0}.pdb'.format(repeat_number)
    dcd_filename = directory + '/out{0}.dcd'.format(repeat_number)

    # Read chemical identities and simulation information:
    ncfile = Dataset(netcdf_filename, 'r')
    identities = ncfile.groups['Sample state data']['identities'][:,:]
    ncfile.close()

    # Load MD trajectory and convert spatial units to Angstroms:
    traj = mdtraj.load(dcd_filename, top=pdb_filename)
    traj.xyz *= 10.0

    return identities, traj

def get_indices(traj):
    """
    Extract the atominc indices for DNA and water oxygens from a simulation.

    Parameters
    ----------
    mdtraj.core.trajectory.Trajectory
        simulation object that contains all residues and coordinates.

    Returns
    -------
    dna_indices: list
        the atomic indices of DNA.
    water_oxygen_indices: list
        the atomic indices of water oxygens.
    """
    # Find water molecules - some of these will have ion non-bonded paramters.
    #waters = [residue for residue in traj.topology.residues if residue.is_water]
    water_oxygen_indices = traj.topology.select_atom_indices('water')

    # Find the DNA residues:
    resnames = ['DA', 'DT', 'DC', 'DG', 'DG3', 'DC5']
    dna_indices = [atom.index for atom in traj.topology.atoms if ((atom.residue.name in resnames))]

    return dna_indices, water_oxygen_indices

def mirror_occupancy_correlation(identities, traj, skip=50, dist_cutoff=5, maxframe=None):
    """
    Get the per frame correlation between the cumulative cation occupancies of the each DNA atom with its
    palindromic copy. Following the analysis of [1], occupancy is defined

        '... using the proximity method (60, 61), in which
        each ion in each MD snapshot is assigned to a DNA atom based
        on closest approach. Ion occupancies for each atom are then
        calculated as ensemble averages of the populations within a
        cutoff distance of 5 Angstroms, chosen as a value that well encompasses
        the first two solvation shells of ions around DNA.'

    [1] S. Y. Ponomarev, K. M. Thayer, D. L. Beveridge, PNAS, 101, 2004, pp. 14771-14775

    Parameters
    ----------
    identities: np.ndarray
        a vector that labels each water molecules as having the nonbonded parameters of either water (0), or cation (1), or anion (2)
    traj: mdtraj.core.trajectory.Trajectory
        simulation object that contains all residues and coordinates.
    target_index: int
        the index of the residues whose interactions with cations will be tracked.
    skip: int
        the frequency with which frames will be analysed. e.g. if skip=2 every second frame will be analysed.
    dist_cutoff: float
        the maximum distance between two atoms that counts as an interaction.
    maxframe: int
        the maximum number of frames that are analysed.

    Returns
    -------
    mirror_cor: numpy.ndarray
        the Pearson correlation coeffiecient of cation occupancies between mirrored DNA atoms for each MD frame.
    """
    [nframes, nwaters] = identities.shape

    if maxframe is None:
        maxframe = nframes

    sq_cutoff = dist_cutoff**2

    dna_indices, water_oxygen_indices = get_indices(traj)

    cation_counts = np.zeros((nframes, len(dna_indices)))

    mirror_cor = []
    for frame in range(0, maxframe, skip):
        cation_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==1]
        dna_coords = traj.xyz[frame, dna_indices, :]
        for cation_index in cation_indices:
            cation_coords = traj.xyz[frame, cation_index, :]
            sq_distances = np.sum((dna_coords  - cation_coords)**2, axis=1)
            min_dist = np.min(sq_distances)
            if min_dist <= sq_cutoff:
                nearest_index = np.where(sq_distances == min_dist)[0][0]
                cation_counts[frame, nearest_index] += 1
        # Count the counts in the forward and reverse direction
        total_counts = cation_counts.sum(axis=0)
        forward_copy_counts = total_counts[0:int(total_counts.shape[0]/2)]
        #mirror_copy_counts  = total_counts[total_counts.shape[0]:int(total_counts.shape[0]/2 - 1):-1]
        # TODO: check that the below correctly matches the reverse of the forward atoms
        mirror_copy_counts  = total_counts[int(total_counts.shape[0]/2):total_counts.shape[0]]
        pearson_corr = np.corrcoef(forward_copy_counts, mirror_copy_counts)[0,1]
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
        mirror_cor.append(pearson_corr)

    return np.array(mirror_cor)

def intersect(a, b):
    return list(set(a) & set(b))

def track_cation_interactions(identities, traj, target_index, dist_cutoff=5, skip=1, maxframe=None):
    """
    Find out which cations are within a specified distance of a specific residue in each frame of the supplied
    trajectory.

    Parameters
    ----------
    identities: np.ndarray
        a vector that labels each water molecules as having the nonbonded parameters of either water (0), or cation (1), or anion (2)
    traj: mdtraj.core.trajectory.Trajectory
        simulation object that contains all residues and coordinates.
    target_index: int
        the index of the residues whose interactions with cations will be tracked.
    dist_cutoff: float
        the maximum distance between two atoms that counts as an interaction.
    skip: int
        the frequency with which frames will be counted. e.g. if skip=2 every second frame will be analysed.
    maxframe: int
        the maximum number of frames that are analysed.

    Return
    ------
    ion_tracker: list
        a list of the cation indices that are within dist_cutoff for each frame.
    """
    [nframes, nwaters] = identities.shape
    if maxframe is None:
        maxframe = nframes

    dna_indices, water_oxygen_indices = get_indices(traj)

    sq_cutoff = dist_cutoff**2

    ion_tracker = []
    for frame in range(0, maxframe, skip):
        cation_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==1]
        target_coords = traj.xyz[frame, target_index, :]
        cation_coords = traj.xyz[frame, cation_indices, :]
        sq_distances = np.sum((cation_coords  - target_coords)**2, axis=1)
        nearest_inds = np.where(sq_distances <= sq_cutoff)
        ion_tracker.append(list(nearest_inds[0]))

    return ion_tracker

def get_autocorrelation(tracker, tmax):
    """
    Generate the contribution to the normalized autocorrelation function for cation interactions with specified DNA atoms.

    Parameters
    ----------
    tracker: list
        Contains the indices of the ions that are within a specified distance of a given atom at each time point.
        Each element of this list is another list, which may be empty (if no ions are within the distance at that
        time point), and can contain multiple elements if multiple ions are within the specified distance.
    tmax: int
        The maximum time point that the contribution to the correlation function will be evaluated.

    Returns
    -------
    c: numpy.ndarray
        the contribution of the ion interactions to the normalized correlation function
    """
    c = np.zeros(tmax + 1)
    for t in range(tmax + 1):
        for t0 in range(len(tracker) - t):
            same_ions = float(len(intersect(tracker[t0], tracker[t0+t])) > 0)
            c[t] += same_ions / (len(tracker) - t)
    return c

def summarize_correlation(data, target_indices, dt, tmax, dist_cutoff=5, skip=1, maxframe=None):
    """
    Extract correlation functions and effective interaction timescales from DNA-ion simulations.

    Parameters
    ----------
    data: tuple
        contains both the
    :param data:
    :param target_indices:
    :param dt:
    :param tmax:
    :param dist_cutoff:
    :param skip:
    :param maxframe:
    :return:
    """
    corr_funcs = []
    timescales = []

    for simulation in data:
        identities, traj = simulation
        for target_index in target_indices:
            tracker = track_cation_interactions(identities, traj, target_index, dist_cutoff, skip, maxframe)
            c = get_autocorrelation(tracker, tmax)
            corr_funcs.append(c)
            timescales.append(np.trapz(c/c[0], dx=dt))
    return corr_funcs, timescales

def bootstrap_correlation_functions(corr_functions, dt, nboots=100):

    corr_boots = np.zeros((nboots, corr_functions.shape[1]))
    timescale_boots = np.zeros(nboots)
    for sample in range(nboots):
        boot_indices = np.random.choice(corr_functions.shape[0], corr_functions.shape[0])
        cb = np.mean(corr_functions[boot_indices, :], axis=0)
        corr_boots[sample, :] = cb / cb[0]
        timescale_boots[sample] = np.trapz( cb / cb[0], dx=dt)

    return corr_boots, timescale_boots
