from netCDF4 import Dataset
import numpy as np
from pymbar import timeseries as ts


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


def read_ncmc_data(folders):
    """
    Read in the data from an NCMC calibration.
    :param folders:
    :return:
    """
    if len(folders) == 0:
        raise Exception('No folders provided')

    nsalt = []
    acceptance_prob = []
    acceptance_error = []
    empirical_accept = []
    log_accept = []
    log_accept_error = []
    npert = []
    time = []

    for folder in folders:
        npert.append(int(folder.split('_')[1]))
        ncfile = Dataset(folder + '/out.nc', 'r')
        naccepted = ncfile.groups['Sample state data']['naccepted'][:]
        # Calculating the acceptance probability directly from the log_accept
        log_acceptance = ncfile.groups['Sample state data']['log_accept'][:]
        mu, sigma, log_mu, log_sigma = calc_acceptance_rate(log_acceptance)
        acceptance_prob.append(mu)
        acceptance_error.append(sigma)
        log_accept.append(log_mu)
        log_accept_error.append(log_sigma)
        # The empirical acceptance probability
        naccepted = naccepted.reshape(naccepted.shape[0])
        empirical_accept.append(naccepted[-1] / len(naccepted))
        # Recording salt number as a function of time in order to estimate the statistical inefficiency.
        nsalt.append(ncfile.groups['Sample state data']['species counts'][:][:, 1])
        # Extracting the mean time for an insertion/deletion in seconds
        time.append(np.mean(ncfile.groups['Sample state data']['time'][:]))
        timestep = float(ncfile.groups['Control parameters']['timestep'][:])  # In femtoseconds
        ncfile.close()

    npert = np.array(npert)
    protocol_length = npert * timestep / 1000.  # The length of the protocol in ps
    time = np.array(time)
    empirical_accept = np.array(empirical_accept)
    log_accept = np.array(log_accept)
    log_accept_error = np.array(log_accept_error)
    acceptance = np.array(acceptance_prob)
    acceptance_error = np.array(acceptance_error)

    return time, acceptance, acceptance_error, log_accept, log_accept_error, empirical_accept, protocol_length


class AutoAnalyzeNCMCOptimization(object):
    def __init__(self, folders):
        if len(folders) == 0:
            raise Exception('No folders provided')

        nsalt = []
        acceptance_prob = []
        acceptance_error = []
        empirical_accept = []
        log_accept = []
        log_accept_error = []
        npert = []
        time = []
        timestep = []

        for folder in folders:
            npert.append(int(folder.split('_')[1]))
            ncfile = Dataset(folder + '/out.nc', 'r')
            naccepted = ncfile.groups['Sample state data']['naccepted'][:]
            # Calculating the acceptance probability directly from the log_accept
            log_acceptance = ncfile.groups['Sample state data']['log_accept'][:]
            mu, sigma, log_mu, log_sigma = calc_acceptance_rate(log_acceptance)
            acceptance_prob.append(mu)
            acceptance_error.append(sigma)
            log_accept.append(log_mu)
            log_accept_error.append(log_sigma)
            # The empirical acceptance probability
            naccepted = naccepted.reshape(naccepted.shape[0])
            empirical_accept.append(naccepted[-1] / len(naccepted))
            # Recording salt number as a function of time in order to estimate the statistical inefficiency.
            nsalt.append(ncfile.groups['Sample state data']['species counts'][:][:, 1])
            # Extracting the mean time for an insertion/deletion in seconds
            time.append(np.mean(ncfile.groups['Sample state data']['time'][:]))
            timestep.append(float(ncfile.groups['Control parameters']['timestep'][:]))  # In femtoseconds
            ncfile.close()

        self.npert = np.array(npert)
        self.timestep = np.array(timestep)
        self.protocol_length = self.npert * self.timestep / 1000.  # The length of the protocol in ps
        self.time = np.array(time)
        self.empirical_accept = np.array(empirical_accept)
        self.log_accept = np.array(log_accept)
        self.log_accept_error = np.array(log_accept_error)
        self.accept = np.array(acceptance_prob)
        self.accept_error = np.array(acceptance_error)
        self.nsalt = np.array(nsalt)

        # Linear fit to estimate wallclock time for number of ncmc perturbation steps.
        self.m, self.c = self.calc_walltime()

    def calc_walltime(self):
        A = np.vstack([self.npert, np.ones(len(self.npert))]).T
        m, c = np.linalg.lstsq(A, self.time)[0]

        return m, c

    def wallclock_time(self, npert):
        """
        The mean wallclock time.
        """
        return self.m * npert + self.c

    def _calc_stat_neff(self):
        """
        Estimate the statistical inefficiency of the salt occupancy.
        """
        stat_ineff = []
        for counts in self.nsalt:
            t, g, Neff = ts.detectEquilibration(counts, fast=True)
            stat_ineff.append(g)
        stat_ineff = np.array(stat_ineff)

        # Correcting the statistical inefficieny has returned a value of 1.0, when there were no acceptances.
        stat_ineff[np.where(stat_ineff == 1.0)] = np.inf

        return stat_ineff

    def calc_efficiency(self, mode='acceptance probability'):

        if mode.lower() in ['acceptance probability', 'acceptance', 'probability']:
            return self.accept/self.wallclock_time(self.npert), self.accept_error/self.wallclock_time(self.npert)
        elif mode.lower() in ['statistical inefficiency', 'statistical']:
            stat_ineff = self._calc_stat_neff()
            return 1. / self.wallclock_time(self.npert) / stat_ineff
