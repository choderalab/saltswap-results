from netCDF4 import Dataset
import numpy as np
from pymbar import bar
import misc_tools
from saltswap.wrappers import Salinator
from pymbar import timeseries

#------ ANALYSIS TOOLS ------#
def read_concentration(files, discard=10, fast=False):
    """
    Calculate the mean concentration and standard error from numerous numerous simulations, where each simulation has
    a fixed chemical potential. Timeseries analysis is used to determine equilibrium properties.

    Parameters
    ----------
    files: list of str
        the path to each results file that will be analysed.
    discard: int
        the initial amount of data to throw away
    fast: bool
        whether to perform the fast varient of the time series analysis
    """
    concentration = np.zeros(len(files))
    standard_error = np.zeros(len(files))
    delta_mu = np.zeros(len(files))
    lower = np.zeros(len(files))
    upper = np.zeros(len(files))
    for i in range(len(files)):
        ncfile = Dataset(files[i], 'r')
        volume = ncfile.groups['Sample state data']['volume'][:]
        #ncations = ncfile.groups['Sample state data']['species counts'][:, 1]
        nsalt = np.min(ncfile.groups['Sample state data']['species counts'][:, 1:2], axis=1)
        delta_mu[i] = ncfile.groups['Control parameters']['delta_chem'][0]
        ncfile.close()

        # Get the concentration in Molarity
        c = 1.0 * nsalt / volume * 1.66054

        # Estimate the mean and standard error with timeseries analysis
        t_equil, stat_ineff, n_eff = timeseries.detectEquilibration(c[discard:], fast=fast)
        #mu, sigma, num_batches, conf_width = misc_tools.batch_estimate_2(c[(discard + t_equil):], stat_ineff)
        #print("{0} batches for {1}".format(num_batches, files[i]))
        c_equil = c[(discard + t_equil):]
        concentration[i] = np.mean(c_equil)
        independent_inds = timeseries.subsampleCorrelatedData(c_equil, g=stat_ineff, conservative=True)
        mu_samps = misc_tools.bootstrap_estimates(c_equil[independent_inds])
        lower[i] = np.percentile(mu_samps, 2.5)
        upper[i] = np.percentile(mu_samps, 97.5)
        standard_error[i] = mu_samps.std()

    return concentration, standard_error, delta_mu, lower, upper


def calc_acceptance_rate(log_accept):
    """
    Calculate the acceptance rate from the log acceptance probability from all proposals, if they were accepted or not

    Parameter
    ---------
    log_accept: numpy.ndarray
        the log of the candidate acceptance probability. Can be greater than zero.

    Returns
    -------
    mu: float
        the mean of the acceptance probability
    sigma: float
        the standard error on the mean of the acceptance probability
    mu_log: float
        the mean of the log acceptance probability.
        Not the same as log(mu) due to Jensen's inequality.
    sigma_log: float
        the standard error on the mean of the log acceptance probability.
        Not the same as log(sigma) due to Jensen's inequality.
    """
    nsamps = len(log_accept)
    # The log acceptance probability
    log_probs = np.min(np.vstack((np.zeros(nsamps), log_accept.reshape(log_accept.shape[0]))), axis=0)
    log_mu = log_probs.mean()
    log_sigma = log_probs.std() / np.sqrt(len(log_probs))
    # The actual acceptance probability
    probs = np.exp(log_probs)
    mu = probs.mean()
    sigma = probs.std() / np.sqrt(len(probs))
    return mu, sigma, log_mu, log_sigma

def calc_relative_free_energy(initial, final, proposals, work, max_waters, multiplicity_correction=True):
    """
    Calculate free energies from protocol work to insert or delete salt for the initial and final number of salt pairs.

    This function searches through all the supplied proposals to find the ones that match the supplied initial and
    final values, and extracts the work for that attempt.


    Assumptions
    -----------
    * There are always equal numbers of cations and anions.
    * The initial and final numbers of salt pairs differ only by 1.
    * The contribution of any changes in volume are encapsulated in the supplies work.

    Parameters
    ----------
    initial: int
        the initial number of salt pairs, from which you want to calculate the free energy from.
    final: int
        the final number of salt pairs, from which you want to calculate the free energy to.
    proposals: numpy.ndarray
        array of the proposed moves, e.g. proposals[10] = np.array([4, 5]).
    work: numpy.ndarray
        array of protocol work for each proposal in kT.
    max_waters: int
        the number of water molecules present before the salt-water exchange.
    multiplicity_correction: bool
        Whether to add the analytical free energy for changing the number of particles in a system.

    Returns
    -------
    estimate: float
        estimate of the free energy in kT.
    error: float
        estimated standard deviation on estimate in kT.
    """
    if abs(initial - final) != 1:
        raise Exception('Only free energies where the initial and final numbers of salt pairs that differ by 1 will be '
                        'computed.')

    work_forward = []
    work_reverse = []
    # Extract the work where the final and initial states of the salt occupancy are as requested.
    for i in range(len(proposals)):
        if np.all(proposals[i] == np.array([initial, final])):
            work_forward.append(work[i])
        elif np.all(proposals[i] == np.array([final, initial])):
            work_reverse.append(work[i])
    work_forward = np.array(work_forward)
    work_reverse = np.array(work_reverse)

    # Use BAR to calculate the free energy difference between the initial and final states
    relative_free_energy, standard_deviation = bar.BAR(work_forward, work_reverse, compute_uncertainty=True)

    # Apply the free energy to change the number of particles in a system.
    if multiplicity_correction:
        nwaters = float(max_waters - initial)
        if initial < final:
            relative_free_energy -= np.log(1.0 * nwaters * (nwaters - 1) / (initial + 1) / (initial + 1))
        else:
            relative_free_energy -= np.log(1.0 * initial * initial / (nwaters + 1) / (nwaters + 2))

    return relative_free_energy, standard_deviation


def calc_volume(volume, nsalt):
    """
    Estimate the volume of the system as a function of number of salt pairs present.

    Parameters
    ----------
    volume: numpy.ndarray
        the volume of the system at each iteration of the simulation.
    nsalt: numpy.ndarray
        the number of salt pairs present at each iteration of the simulation.

    Returns
    -------
    average: numpy.ndarray
        The mean volume of the system at each salt occupancy.
    error: numpy.ndarray
        The standard error on the mean volume.
    """
    max_salt = int(np.max(nsalt))
    average = np.zeros(max_salt + 1)
    error = np.zeros(max_salt + 1)

    for n in range(max_salt + 1):
        voln = volume[np.where(nsalt == n)]
        average[n] = voln.mean()
        error[n] = voln.std() / np.sqrt(len(voln))

    return average, error

def calc_concentration(volume, nsalt):
    """
    Calculate the concentration over the course of a simulation.

    Parameters
    ----------
    volume: numpy.ndarray
        the instantaneous volume over the course of a simulation in nm**3
    nsalt: numpy.ndarray
        the instantaneous number of salt pairs during a simulation.
    """

    return nsalt/volume * 1.66054

def predict_ensemble_concentration(delta_chem, relative_free_energy, std_free, volume, std_volume, nsamples=500):
    """
    Calculate the average concentration of salt (in M) at a given chemical potential for multiple bootstrap samples of
    the relative free energies between salt occupancies. Useful for estimating the uncertainty in predicted
    concentrations.

    Parameters
    ----------
    delta_chem: float
        The difference between the chemical potential of two water molecules and anion and cation (in kT).
    relative_free_energy: numpy.ndarray
        The relative free energy to add salt for consecutive numbers of salt (in kT).
    std_free: numpy.ndarray
        The standard deviation of the uncertainty on the relative free energies.
    volume: numpy.ndarray
        The mean volume (in nm**3) as a function of the number of salt pairs.
    std_volume: numpy.ndarray
        The standard error on the mean volume.
    nsamples: int
        The number of bootstrap samples to take

    Returns
    -------
    concentration: numpy.ndarray
        The mean concentration (in mols/litre) of salt for each bootstrap sample.
    """
    nsalt = np.arange(0, len(volume))
    concentration = np.zeros(nsamples)

    for sample in range(nsamples):
        energy_samples = np.random.normal(loc=relative_free_energy, scale=std_free)
        volume_samples = np.random.normal(loc=volume, scale=std_volume)
        cumulative_free_energy = np.hstack((0.0, np.cumsum(energy_samples)))
        exponents = -delta_chem * nsalt - cumulative_free_energy
        a = np.max(exponents)
        numerator = np.sum(nsalt * np.exp(exponents - a))
        denominator = np.sum(volume_samples * np.exp(exponents - a))
        concentration[sample] = numerator/denominator * 1.66054
    return concentration

class AutoAnalyzeCalibration(object):
    """
    Reads in self-adjusted mixture sampling saltswap data and automates the calibration of the chemical potential.
    """
    def __init__(self, files):
        """
        Read in simulation data and runs through the calibration.

        Parameters
        ----------
        files: list of str
            the list of the NetCDF files containing the SAMS simulation data
        """

        # Read in the files
        self.files = files
        self.volume, self.nwats, self.nsalt, self.work, self.proposal, self.sams = self._read_files()

        # The free energies calculated with BAR
        self.relative_free_energy, self.error_free_energy = self._get_BAR_energies()

        # Use the relative free energies to get the cumulative free energies.
        self.cumulative_free_energy = np.hstack((0.0,np.cumsum(self.relative_free_energy)))

        # Calculate the volume as a function of salt occupancy
        self.average_volume, self.error_volume = calc_volume(self.volume, self.nsalt)

    def _read_files(self):
        """
        Load the data for the calibration.
        """
        volume = []
        sams_estimate = []
        nwats = []
        nsalt = []
        proposal = []
        protocol_work = []
        for file in self.files:
            ncfile = Dataset(file, 'r')
            volume.append(ncfile.groups['Sample state data']['volume'][:])
            sams_estimate.append(ncfile.groups['Sample state data']['sams bias'][-1, :])
            proposal.append(ncfile.groups['Sample state data']['proposal'][:][:, 0, :])
            nspecies = ncfile.groups['Sample state data']['species counts']
            protocol_work.append(ncfile.groups['Sample state data']['cumulative work'][:, 0, -1])
            nwats.append(nspecies[:, 0])
            nsalt.append(nspecies[:, 1])
            ncfile.close()

        volume = np.hstack(volume)
        nwats = np.hstack(nwats)
        nsalt = np.hstack(nsalt)
        protocol_work = np.hstack(protocol_work)
        proposal = np.vstack(proposal)

        return volume, nwats, nsalt, protocol_work, proposal, sams_estimate

    def _get_BAR_energies(self):
        """
        Use BAR to calculate the free energy to add salt for all salt pairs in the simulation

        Returns
        -------
        relative_free_energy: np.ndarray
            The relative free energy to add salt for all consectutive pairs.
        error: numpy.ndarray
            The standard erorr of the free energy estimates.

        """
        max_nsalt = int(np.max(self.nsalt))
        relative_free_energy = np.zeros(max_nsalt)
        error = np.zeros(max_nsalt)
        for i in range(max_nsalt):
            f, e = calc_relative_free_energy(i, i + 1, max_waters=np.max(self.nwats), proposals=self.proposal,
                                            work=self.work, multiplicity_correction=True)
            relative_free_energy[i] = f
            error[i] = e

        return relative_free_energy, error

    def predict_ensemble_concentrations(self, deltachems, nsamples):
        """
        Calculate the macroscopic concentration of salt (in M) and bootstrap samples given the volume and relative free
        energy per salt pair.

        Parameters
        ----------
        deltachems: numpy.ndarray
            Array of chemical potentials for which the macroscopic concentration will be predicted.
        nsamples: int
            The number of bootstrap samples used for the concentration error distribution.

        Returns
        -------
        mean_concentration: numpy.ndarray
            the maximum likelihood estimate of the macroscopic concentration at each chemical potential.
        ensemble_concentration: numpy.ndarray
            Estimates of the mean concentration for different bootstrap samples of the free energies and volumes.
        """
        ensemble_concentration = np.zeros((len(deltachems), nsamples))
        mean_concentration = np.zeros(len(deltachems))
        for i in range(len(deltachems)):
            mean_concentration[i] = Salinator.predict_concentration(deltachems[i], self.cumulative_free_energy,
                                                                    self.average_volume)
            ensemble_concentration[i, :] = predict_ensemble_concentration(deltachems[i], nsamples=nsamples,
                                                                          volume=self.average_volume,
                                                                          std_volume=self.error_volume,
                                                                          relative_free_energy=self.relative_free_energy,
                                                                          std_free=self.error_free_energy)
        return mean_concentration, ensemble_concentration
