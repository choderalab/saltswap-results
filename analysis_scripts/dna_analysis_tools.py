import numpy as np
from netCDF4 import Dataset
import mdtraj
from time import time

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
    #traj.xyz *= 10.0

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


#----------- Functions for left-right correlation analysis -----------#
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
    nwaters = identities.shape[1]
    nframes = len(traj)

    if maxframe is None:
        maxframe = nframes


    dna_indices, water_oxygen_indices = get_indices(traj)
    frames = list(range(0, maxframe, skip))
    cation_counts = np.zeros((len(frames), len(dna_indices)))

    mirror_cor = []
    for f in range(len(frames)):
        cation_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frames[f], water_index]==1]
        traj_frame = traj[frames[f]]
        for cation_index in cation_indices:
            pairs = np.vstack((dna_indices, np.repeat(cation_index, len(dna_indices)))).T
            dists = mdtraj.compute_distances(traj_frame, pairs, periodic=True, opt=True)[0] * 10.0
            min_dist = np.min(dists)
            if min_dist <= dist_cutoff:
                nearest_index = np.where(dists == min_dist)[0][0]
                cation_counts[f, nearest_index] += 1
        # Count the counts in the forward and reverse direction
        total_counts = cation_counts.sum(axis=0)
        forward_copy_counts = total_counts[0:int(total_counts.shape[0]/2)]
        mirror_copy_counts  = total_counts[int(total_counts.shape[0]/2):total_counts.shape[0]]
        pearson_corr = np.corrcoef(forward_copy_counts, mirror_copy_counts)[0,1]
        if np.isnan(pearson_corr):
            pearson_corr = 0.0
        mirror_cor.append(pearson_corr)

    return np.array(mirror_cor)

#----------- Functions for ion-phosphate autocorrelation analysis -----------#

def intersect(a, b):
    return list(set(a) & set(b))


def autocorrelation_time(c, dt):
    """
    For an estimated autocorrelation function, estimate the autocorrelation time.

    Reference
    ---------
    W. Janke, "Statistical Analysis of Simulations: Data Correlations and Error Estimation", published in
    "Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms", Lecture Notes,
    J. Grotendorst, D. Marx, A. Muramatsu (Eds.), John von Neumann Institute for Computing, Vol. 10,pp. 423-445, 2002.

    Parameters
    ----------
    c: numpy.ndarray
        the autocorrelation function. Doesn't need to be normalized.
    dt: float
        the amount of time elapasesed between the indices of c.

    Returns
    -------
    t_auto_corr: float
        the autocorrelation time
    t_exponential: float
        the exponential autocorrelation time. If the autocorrelation function is a weighted sum exponential functions,
        then t_exponential is the weighted mean of the decay times.
    """
    # Normalize:
    c_norm = c/c[0]

    k = np.arange(0, len(c))
    t_auto_corr = (0.5 + np.sum(c_norm - (c_norm * k) / len(c))) * dt
    t_exponential = np.trapz(c_norm, dx=dt) # The area under the curve

    return t_auto_corr, t_exponential


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
    nwaters = identities.shape[1]
    nframes = len(traj)

    if maxframe is None:
        maxframe = nframes

    dna_indices, water_oxygen_indices = get_indices(traj)

    ion_tracker = []
    for frame in range(0, maxframe, skip):
        traj_frame = traj[frame]
        cation_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==1]
        pairs = np.vstack((cation_indices, np.repeat(target_index, len(cation_indices)))).T
        dists = mdtraj.compute_distances(traj_frame, pairs, periodic=True, opt=True)[0] * 10.0 # Converting to Angs.
        nearest_inds = np.where(dists <= dist_cutoff)
        ion_tracker.append(list(nearest_inds[0]))

    return ion_tracker


def get_autocorrelation(tracker, tmax):
    """
    Generate the autocovariance of ion occupancies as specified in the supplied ion tracker.

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
        contains the saltwap "identities" array and MD data stored as an mdtraj.traj object.
    dt: float
        the conversion factor between index and time.
    tmax: int
        The maximum time point that the contribution to the correlation function will be evaluated.
    target_index: int
        the index of the residues whose interactions with cations will be tracked.
    dist_cutoff: float
        the maximum distance between two atoms that counts as an interaction.
    skip: int
        the frequency with which frames will be counted. e.g. if skip=2 every second frame will be analysed.
    maxframe: int
        the maximum number of frames that are analysed.

    Returns
    -------
    corr_funcs: list of numpy.ndarray
        the correlation functions of ion occupancies
    timescales: list of floats
        the autocorrelation times
    """
    corr_funcs = []
    auto_timescales = []
    expo_timescales = []

    for simulation in data:
        identities, traj = simulation
        for target_index in target_indices:
            tracker = track_cation_interactions(identities, traj, target_index, dist_cutoff, skip, maxframe)
            c = get_autocorrelation(tracker, tmax)
            corr_funcs.append(c)
            tau_autocorr, tau_exponential = autocorrelation_time(c, dt)
            auto_timescales.append(tau_autocorr)
            expo_timescales.append(tau_exponential)
    return np.array(corr_funcs), np.array(auto_timescales), np.array(expo_timescales)

def bootstrap_correlation_functions(corr_functions, dt, nboots=1000):

    corr_boots = np.zeros((nboots, corr_functions.shape[1]))
    auto_timescales = np.zeros(nboots)
    expo_timescales = np.zeros(nboots)
    for sample in range(nboots):
        boot_indices = np.random.choice(corr_functions.shape[0], corr_functions.shape[0])
        cb = np.mean(corr_functions[boot_indices, :], axis=0)
        corr_boots[sample, :] = cb / cb[0]
        tau_autocorr, tau_exponential = autocorrelation_time(cb, dt)
        auto_timescales[sample] = tau_autocorr
        expo_timescales[sample] = tau_exponential

    return corr_boots, auto_timescales, expo_timescales

#----------- Functions for charge spatial distribution analysis -----------#

def get_ion_distributions(traj, identities, distances, solute_indices, water_oxygen_indices, min_frame=0, skip=10):
    """
    Get samples of the number of cations, anions, and salt pairs contained within different minimum distances away from a solute.

    Parameters
    ----------
    traj: mdtraj.core.trajectory.Trajectory
        simulation object that contains all residues and coordinates.
    identities: np.ndarray
        a vector that labels each water molecules as having the nonbonded parameters of either water (0), or cation (1),
        or anion (2).
    distances: numpy.ndarray
        Vector of distances within which the number of salt pairs will be counted.
    solute_indices: numpy.ndarray
        the mdtraj indices of the solute atoms
    water_oxygen_indices: numpy.ndarray
        the mdtraj indices of the water oxygen atoms
    min_frame: int
        the minimum frame from which to start the analysis
    skip: int
        the number of frames to skip in the analysis.

    Returns
    -------
    nsalt_at_dist: numpy.ndarray
        the number of salt pairs within the supplied distances at each frame.
    ncations_at_dist: numpy.ndarray
        the number of cations within the supplied distances at each frame.
    nanions_at_dist: numpy.ndarray
        the number of anions within the supplied distances at each frame.
    """
    nwaters = identities.shape[1]
    nframes = len(traj)

    frames = list(range(min_frame, nframes, skip))
    nsalt_at_dist = np.zeros((len(frames), len(distances)))
    ncations_at_dist = np.zeros((len(frames), len(distances)))
    nanions_at_dist = np.zeros((len(frames), len(distances)))

    t0 = time()   # The below takes around 0.414 seconds per frame.
    for f in range(len(frames)):
        # Get indices
        cation_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frames[f], water_index]==1]
        anion_indices = [water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frames[f], water_index]==2]

        # Get distances
        traj_frame = traj[frames[f]]
        cation_dists = np.zeros(len(cation_indices))
        for ion_ind in range(len(cation_indices)):
            pairs = np.vstack((solute_indices, np.repeat(cation_indices[ion_ind], len(solute_indices)))).T
            cation_dists[ion_ind] = mdtraj.compute_distances(traj_frame, pairs, periodic=True, opt=True).min() * 10.0

        anion_dists = np.zeros(len(anion_indices))
        for ion_ind in range(len(anion_indices)):
            pairs = np.vstack((solute_indices, np.repeat(anion_indices[ion_ind], len(solute_indices)))).T
            anion_dists[ion_ind] = mdtraj.compute_distances(traj_frame, pairs, periodic=True, opt=True).min() * 10.0

        for d in range(len(distances)):
            ncations_at_dist[f, d] = np.sum(cation_dists <= distances[d])
            nanions_at_dist[f, d] = np.sum(anion_dists <= distances[d])
            nsalt_at_dist[f, d] = min(ncations_at_dist[f, d], nanions_at_dist[f, d])

    print('\nTime taken to get ion distributions = {0:.2f} seconds.'.format(time() - t0))

    return nsalt_at_dist, ncations_at_dist, nanions_at_dist

def wrapper_ion_distance_profile(directory, repeats, distances, skip, min_frame):
    """
    A wrapper function to load DNA simulation data and provide estimates of the charge RDF.
    """
    # Load simulations
    sim_data = []    # list of (identities, traj)
    for repeat in repeats:
        sim_data.append(load_simulation(directory, repeat))
    print('Simuluation data loaded')

    dna_indices, water_oxygen_indices = get_indices(sim_data[0][1])

    nsalt_at_dist = []
    ncations_at_dist = []
    nanions_at_dist = []
    # Get samples of charge at different radii from DNA
    for r in range(len(repeats)):
        s, c, a = get_ion_distributions(sim_data[r][1], sim_data[r][0], distances, dna_indices, water_oxygen_indices, min_frame=min_frame, skip=skip)
        nsalt_at_dist.append(s)
        ncations_at_dist.append(c)
        nanions_at_dist.append(a)

    return np.vstack(nsalt_at_dist), np.vstack(ncations_at_dist), np.vstack(nanions_at_dist)
