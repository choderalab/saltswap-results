import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

from pymbar import timeseries

# Plotting parameters
FONTSIZE = 15
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
files = ['../dhfr/100mM/' + f for f in file_names]
conc_100mM = read_concentration(files)

file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../dhfr/200mM/' + f for f in file_names]
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

corr_t_200mM = 0
for i in range(runs):
    t, g, Neff = timeseries.detectEquilibration(conc_200mM[i,:])
    corr_t_200mM += (g-1)/2.0
corr_t_200mM = corr_t_200mM/float(runs)

corfunc_100mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_100mM[0,:],conc_100mM[1,:],conc_100mM[2,:])), norm=True)
corfunc_200mM = timeseries.normalizedFluctuationCorrelationFunction(np.hstack((conc_200mM[0,:],conc_200mM[1,:],conc_200mM[2,:])), norm=True)

fig, ax = plt.subplots(2, 2, figsize=(11,5),gridspec_kw = {'width_ratios':[3, 1]})

t_sim = np.arange(len(conc_100mM[i,:])) * iter2ns
max_corrfunc_iter = 100
t_corr = np.arange(max_corrfunc_iter) * iter2ns

concs = [conc_100mM[0,:], conc_200mM[0,:]]
corr_funcs = [corfunc_100mM[0:max_corrfunc_iter], corfunc_200mM[0:max_corrfunc_iter]]
auto_corr_time = [corr_t_100mM*iter2ns, corr_t_200mM*iter2ns]

colors = ['C2', 'C4']
axis_nudge = 0.01

for i in range(len(concs)):
    ax[i, 0].plot(t_sim, concs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 0].set_ylim(np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge)
    ax[i, 0].grid(alpha=0.5)
for j in range(2):
    ax[j, 1].plot(t_corr, corr_funcs[j], color=colors[j], lw=LINEWIDTH)
    ax[j, 1].set_ylim((-0.15, 1.05))
    ax[j, 1].axhline(0, color='grey', ls='--')

fig.text(0.06, 0.5, 'Concentration ($M$)', va='center', rotation='vertical', fontsize=FONTSIZE)
fig.text(0.67, 0.5, 'Autocorrelation', va='center', rotation='vertical', fontsize=FONTSIZE)

fig.text(0.15, 0.83, '$100mM$', fontsize=FONTSIZE-1)
fig.text(0.15, 0.41, '$200mM$', fontsize=FONTSIZE-1)

fig.text(0.8, 0.8, '$\\tau$ = {0:.2f}'.format(corr_t_100mM*iter2ns), fontsize=FONTSIZE-1)
fig.text(0.8, 0.38, '$\\tau$ = {0:.2f}'.format(corr_t_200mM*iter2ns), fontsize=FONTSIZE-1)

ax[1, 0].set_xlabel('Time (ns)', fontsize=FONTSIZE)
ax[1, 1].set_xlabel('Time (ns)', fontsize=FONTSIZE)

plt.savefig('DHFR_sampling.png', dpi=DPI)
