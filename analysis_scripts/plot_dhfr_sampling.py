import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from pymbar import timeseries

# Plotting parameters
FONTSIZE = 14
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

# The conversion factor from iterations to nanoseconds
iter2ns = 4*2000*2E-6

#Calculating the mean correlation time from the runs
runs = 3
corr_t_100mM = 0
for i in range(runs):
    t, g, Neff = timeseries.detectEquilibration(conc_100mM[i,:])
    corr_t_100mM += (g-1)/2.0
corr_t_100mM = corr_t_100mM/float(runs)

corr_t_150mM = 0
for i in range(runs):
    t, g, Neff = timeseries.detectEquilibration(conc_150mM[i,:])
    corr_t_150mM += (g-1)/2.0
corr_t_150mM = corr_t_150mM/float(runs)


corr_t_200mM = 0
for i in range(runs):
    t, g, Neff = timeseries.detectEquilibration(conc_200mM[i,:])
    corr_t_200mM += (g-1)/2.0
corr_t_200mM = corr_t_200mM/float(runs)

corfunc_100mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_100mM[0,:],conc_100mM[1,:],conc_100mM[2,:])), norm=True)
corfunc_150mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_150mM[0,:],conc_150mM[1,:],conc_150mM[2,:])), norm=True)
corfunc_200mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_200mM[0,:],conc_200mM[1,:],conc_200mM[2,:])), norm=True)
### Creating a grid of subplots with specific spacing
plt.figure(figsize=(10, 6))
gs1 = gridspec.GridSpec(3, 6)
gs1.update(top=0.98, bottom= 0.08, left=0.08, right=0.7, wspace=0.0, hspace=0.3)
gs2 = gridspec.GridSpec(3, 1)
gs2.update(top=0.98, bottom= 0.08, left=0.77, right=0.98, hspace=0.3)

ax = np.zeros((3,3), dtype=object )
for i in range(3):
    ax[i, 0] = plt.subplot(gs1[i, :5])
    ax[i, 1] = plt.subplot(gs1[i, 5:6])
    ax[i, 2] = plt.subplot(gs2[i,:])

# Setting the length of the xaxes
t_sim = np.arange(len(conc_100mM[0,:])) * iter2ns
max_corrfunc_iter = 220
t_corr = np.arange(max_corrfunc_iter) * iter2ns

simulation = 2
concs = [conc_100mM[simulation,:], conc_150mM[simulation,:], conc_200mM[simulation,:]]
corr_funcs = [corfunc_100mM[0:max_corrfunc_iter], corfunc_150mM[0:max_corrfunc_iter], corfunc_200mM[0:max_corrfunc_iter]]
auto_corr_time = [corr_t_100mM*iter2ns, corr_t_150mM*iter2ns, corr_t_200mM*iter2ns]

colors = ['C2', 'C0', 'C4']
axis_nudge = 0.01

s_timeseries = ['$0.1M$', '$0.15M$', '$0.2M$']
YLIM = (np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge*6)
t_dens = np.linspace(YLIM[0], YLIM[1])

BW = 0.3
for i in range(len(concs)):
    # Timeseries
    ax[i, 0].plot(t_sim, concs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 0].set_ylim(YLIM)
    ax[i, 0].grid(alpha=0.5, ls='--')
    ax[i, 0].set_xlim(0,np.max(t_sim))
    ax[i, 0].set_ylabel('Concentration ($M$)', fontsize=FONTSIZE)
    ax[i, 0].text(0.4, 0.27, 'Macroscopic concentration =' + s_timeseries[i], fontsize=FONTSIZE)
    # Histograms
    #ax[i, 1].hist(concs[i], orientation='horizontal', color=colors[i], bins=8)
    #ax[i, 1].set_ylim(np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge)
    #ax[i, 1].set_axis_off()
    # Kernel density estimate
    dens = gaussian_kde(concs[i], bw_method=BW)
    t_dens = np.linspace(np.min(concs[i]), np.max(concs[i]))
    ax[i, 1].fill_between(dens(t_dens), t_dens, color=colors[i])
    ax[i, 1].set_ylim(YLIM)
    ax[i, 1].set_axis_off()
    # Autocorrelation function
    ax[i, 2].plot(t_corr, corr_funcs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 2].set_ylim((-0.15, 1.05))
    ax[i, 2].axhline(0, color='grey', ls='--')
    ax[i, 2].set_ylabel('Autocorrelation', fontsize=FONTSIZE)
    ax[i, 2].text(0.5, 0.85, '$\\tau$ = {0:.2f}'.format(corr_t_100mM*iter2ns), fontsize=FONTSIZE)

ax[2, 0].set_xlabel('Time (ns)', fontsize=FONTSIZE)
ax[2, 2].set_xlabel('Time (ns)', fontsize=FONTSIZE)

plt.savefig('DHFR_sampling.png', dpi=DPI)
