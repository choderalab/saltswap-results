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

### The actual plotting part
fig, ax = plt.subplots(3, 2, figsize=(11,8), gridspec_kw={'width_ratios':[3, 1]})
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.07)



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

for i in range(len(concs)):
    ax[i, 0].plot(t_sim, concs[i], color=colors[i], lw=LINEWIDTH)
    ax[i, 0].set_ylim(np.min(concs[0])-axis_nudge, np.max(concs[-1])+axis_nudge)
    ax[i, 0].grid(alpha=0.5)
for j in range(len(concs)):
    ax[j, 1].plot(t_corr, corr_funcs[j], color=colors[j], lw=LINEWIDTH)
    ax[j, 1].set_ylim((-0.15, 1.05))
    ax[j, 1].axhline(0, color='grey', ls='--')

fig.text(0.015, 0.5, 'Concentration ($M$)', va='center', rotation='vertical', fontsize=FONTSIZE)
fig.text(0.7, 0.5, 'Autocorrelation', va='center', rotation='vertical', fontsize=FONTSIZE)

# The text heights are chosen to all be the same distance from each other and correctly aligned to plots
text_heights = 0.9 - 0.31*np.arange(3)

fig.text(0.1, text_heights[0], '$100mM$', fontsize=FONTSIZE)
fig.text(0.1, text_heights[1], '$150mM$', fontsize=FONTSIZE)
fig.text(0.1, text_heights[2], '$200mM$', fontsize=FONTSIZE)

fig.text(0.8, text_heights[0], '$\\tau$ = {0:.2f}'.format(corr_t_100mM*iter2ns), fontsize=FONTSIZE)
fig.text(0.8, text_heights[1], '$\\tau$ = {0:.2f}'.format(corr_t_150mM*iter2ns), fontsize=FONTSIZE)
fig.text(0.8, text_heights[2], '$\\tau$ = {0:.2f}'.format(corr_t_200mM*iter2ns), fontsize=FONTSIZE)

ax[2, 0].set_xlabel('Time (ns)', fontsize=FONTSIZE)
ax[2, 1].set_xlabel('Time (ns)', fontsize=FONTSIZE)

plt.savefig('DHFR_sampling.png', dpi=DPI)
