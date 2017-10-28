import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pymbar import timeseries

# Plotting parameters
FONTSIZE = 13
TICKSIZE = 12
LEGENDSIZE = 12
LINEWIDTH = 2
DPI = 300

# Read in data
def read_concentration(files):
    """
    Extract the salt concentration from a number of simulation data files.
    """
    nsalt = []
    volume = []
    for file in files:
        ncfile = Dataset(file,'r')
        volume.append(ncfile.groups['Sample state data']['volume'][:])
        nspecies = ncfile.groups['Sample state data']['species counts'][:]
        nanion = nspecies[:,2]
        nsalt.append(nanion)
        ncfile.close()
    volume = np.array(volume)
    nsalt = np.array(nsalt)
    concentration = 1.0*nsalt/volume * 1.66054 # in M

    return concentration

file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/dhfr/100mM/' + f for f in file_names]
conc_100mM = read_concentration(files)

file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/dhfr/150mM/' + f for f in file_names]
conc_150mM = read_concentration(files)

file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/dhfr/200mM/' + f for f in file_names]
conc_200mM = read_concentration(files)

# The conversion factor from iterations to nanoseconds.
iter2ns = 4*2000*2E-6
# Explaination: as stated in ../testsystems/dhfr/100mM/submit_template.lsf, the simulations data was saved every 4
# iterations, where each step took 2000 MD steps with 2 fs (2E-6 ns) timestep.


#Calculating the mean correlation time from the runs
def summarize_timeseries(concentrations):
    """
    Use boostrap sampling together with statisticalInefficiencyMultiple to calculate the mean correlation function and
    its 95% confidence intervals as well as the estimated autocorrelation time.

    Parameters
    ----------
    concentrations: np.ndarray
        Arrays with different time series data in each row.

    Returns
    -------
    mean_corr_func, lower, upper: numpy.ndarray
        the mean, 5th percentile, and 97.5 percentile of the estimated autocorrelation function.
    auto_corr_time_mean, auto_corr_time_std
        the mean autocorrelation time with its standard error.
    """
    boot_samples = 50
    auto_corr_time = np.zeros(boot_samples)
    corr_data = []
    for sample in range(boot_samples):
        ints = np.random.choice(3, 3)
        concs = [concentrations[ints[0],:], concentrations[ints[1],:], concentrations[ints[2],:]]
        g, c = timeseries.statisticalInefficiencyMultiple(concs, return_correlation_function=True, fast=False)
        auto_corr_time[sample] = (g-1)/2.0
        corr_data += c

    # The correlation time may be computed up to a different maximum time for different bootstrap samples.
    # Finding the maximum time.
    max_time = 0
    for tup in corr_data:
        if tup[0] > max_time:
            max_time = tup[0]

    # Unpacking each bootstrapped correlation function for easier analysis
    unpacked_corr_func = {}
    for i in range(max_time):
        unpacked_corr_func[i] = []

    for data in corr_data:
        unpacked_corr_func[data[0]-1].append(data[1])

    # Working out the confidense intervals.
    mean_corr_func = np.zeros(max_time)
    lower = np.zeros(max_time)
    upper = np.zeros(max_time)
    for i in range(max_time):
        mean_corr_func[i] = np.mean(unpacked_corr_func[i])
        lower[i] = np.percentile(unpacked_corr_func[i], q=2.5)
        upper[i] = np.percentile(unpacked_corr_func[i], q=97.5)

    # When the first lower estimate hits zero, ensure that all supsequent data points are also zero.
    zero_from = np.where(lower <= 0.0)[0][0]
    lower[zero_from:] = 0

    return mean_corr_func, lower, upper, auto_corr_time.mean(), auto_corr_time.std()

corfunc_100mM, corfunc_100mM_lower, corfunc_100mM_upper, corr_t_100mM, corr_t_100mM_std = summarize_timeseries(conc_100mM)
corfunc_150mM, corfunc_150mM_lower, corfunc_150mM_upper, corr_t_150mM, corr_t_150mM_std = summarize_timeseries(conc_150mM)
corfunc_200mM, corfunc_200mM_lower, corfunc_200mM_upper, corr_t_200mM, corr_t_200mM_std = summarize_timeseries(conc_200mM)

#runs = 3
#corr_t_100mM = 0
#for i in range(runs):
#    t, g, Neff = timeseries.detectEquilibration(conc_100mM[i,:])
#    corr_t_100mM += (g-1)/2.0
#corr_t_100mM = corr_t_100mM/float(runs)

#corr_t_150mM = 0
#for i in range(runs):
#    t, g, Neff = timeseries.detectEquilibration(conc_150mM[i,:])
#    corr_t_150mM += (g-1)/2.0
#corr_t_150mM = corr_t_150mM/float(runs)


#corr_t_200mM = 0
#for i in range(runs):
#    t, g, Neff = timeseries.detectEquilibration(conc_200mM[i,:])
#    corr_t_200mM += (g-1)/2.0
#corr_t_200mM = corr_t_200mM/float(runs)

#corfunc_100mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_100mM[0,:],conc_100mM[1,:],conc_100mM[2,:])), norm=True)
#corfunc_150mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_150mM[0,:],conc_150mM[1,:],conc_150mM[2,:])), norm=True)
#corfunc_200mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_200mM[0,:],conc_200mM[1,:],conc_200mM[2,:])), norm=True)

### Creating a grid of subplots with specific spacing
plt.figure(figsize=(10, 6))
gs1 = gridspec.GridSpec(3, 6)
gs1.update(top=0.96, bottom= 0.08, left=0.08, right=0.7, wspace=0.0, hspace=0.3)
gs2 = gridspec.GridSpec(3, 1)
gs2.update(top=0.96, bottom= 0.08, left=0.77, right=0.98, hspace=0.3)

ax = np.zeros((3,3), dtype=object )
for i in range(3):
    ax[i, 0] = plt.subplot(gs1[i, :5])
    ax[i, 1] = plt.subplot(gs1[i, 5:6])
    ax[i, 2] = plt.subplot(gs2[i,:])

# Setting the length of the xaxes
t_sim = np.arange(len(conc_100mM[0,:])) * iter2ns
max_corrfunc_iter = max(len(corfunc_100mM_upper), len(corfunc_150mM_upper), len(corfunc_200mM_upper))
t_corr = np.arange(max_corrfunc_iter) * iter2ns

simulation = 2
flat_concs = [conc_100mM.flatten() * 1000, conc_150mM.flatten() * 1000, conc_200mM.flatten() * 1000]
concs = [conc_100mM[simulation, :]*1000, conc_150mM[simulation, :]*1000, conc_200mM[simulation, :]*1000]
corr_funcs = [corfunc_100mM, corfunc_150mM, corfunc_200mM]
corr_funcs_lower = [corfunc_100mM_lower, corfunc_150mM_lower, corfunc_200mM_lower]
corr_funcs_upper = [corfunc_100mM_upper, corfunc_150mM_upper, corfunc_200mM_upper]
auto_corr_time = [corr_t_100mM * iter2ns, corr_t_150mM * iter2ns, corr_t_200mM * iter2ns]
auto_corr_time_std = [corr_t_100mM_std * iter2ns, corr_t_150mM_std * iter2ns, corr_t_200mM_std * iter2ns]

colors = ['C2', 'C0', 'C4']
axis_nudge = 10.0

s_timeseries = ['100mM', '150mM', '200mM']
YLIM = (np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge*6)
t_dens = np.linspace(YLIM[0], YLIM[1])

BW = 0.2
for i in range(len(concs)):
    # Timeseries
    ax[i, 0].plot(t_sim, concs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 0].set_ylim(YLIM)
    ax[i, 0].grid(alpha=0.5, ls='--')
    ax[i, 0].set_xlim(0,np.max(t_sim) + 0.1)
    ax[i, 0].set_ylabel('Concentration $c$ (mM)', fontsize=FONTSIZE - 1)
    ax[i, 0].text(0.4, 270, 'Macroscopic concentration =' + s_timeseries[i], fontsize=FONTSIZE)
    # Histograms
    #ax[i, 1].hist(concs[i], orientation='horizontal', color=colors[i], bins=8)
    #ax[i, 1].set_ylim(np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge)
    #ax[i, 1].set_axis_off()
    # Kernel density estimate using all repeats at a fixed macro concentration.
    #dens = gaussian_kde(concs[i], bw_method=BW)
    dens = gaussian_kde(flat_concs[i], bw_method=BW)
    conc_values = np.linspace(np.min(concs[i]), np.max(concs[i]))
    ax[i, 1].fill_between(dens(conc_values), conc_values, color=colors[i])
    ax[i, 1].set_ylim(YLIM)
    ax[i, 1].set_axis_off()
    # Autocorrelation function
    t_corr = np.arange(len(corr_funcs[i])) * iter2ns
    ax[i, 2].plot(t_corr, corr_funcs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 2].fill_between(t_corr, corr_funcs_lower[i], corr_funcs_upper[i], alpha=0.3, color=colors[i], lw=0)
    ax[i, 2].set_ylim((-0.15, 1.05))
    ax[i, 2].set_xlim(0, 3.5)
    ax[i, 2].axhline(0, color='grey', ls='--')
    ax[i, 2].set_ylabel('Autocorrelation', fontsize=FONTSIZE)
    ax[i, 2].text(0.5, 0.85, '$\\tau$ = {0:.2f} $\pm$ {1:.2f} ns'.format(auto_corr_time[i], auto_corr_time_std[i] * 2.), fontsize=FONTSIZE)

ax[2, 0].set_xlabel('Time (ns)', fontsize=FONTSIZE)
ax[2, 2].set_xlabel('Time (ns)', fontsize=FONTSIZE)

plt.savefig('DHFR_sampling.png', dpi=DPI)
