import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import dna_analysis_tools as tools
from time import time

# Load simulation data
repeat_numbers = [1, 2, 3]

# The list below will hold the simulation data for all the repeats.
fluc_data = []

for repeat in repeat_numbers:
    directory = '../testsystems/dna_dodecamer/200mM'
    fluc_data.append(tools.load_simulation(directory, repeat))
    print('Saltswap repeat {0} loaded'.format(repeat))

fixed_data = []
for repeat in repeat_numbers:
    directory = '../testsystems/dna_dodecamer/200mM_fixed_number'
    fixed_data.append(tools.load_simulation(directory, repeat))
    print('Fixed salt repeat {0} loaded'.format(repeat))


# Find the phosphorous atoms from the adenosine phosphates
ap_indices = [atom.index for atom in fluc_data[0][1].topology.atoms if ((atom.residue.name == 'DA') and atom.name == 'P')]

SKIP = 1
MAXFRAME = 10000 # The number of simulation frames that will be analysed. The larger this is, the less noisy the correlation function.
TMAX = 1250 # The number of frames up to which the correlation function will be analysed.
CUTOFF = 5
iter2ns = 2000 * SKIP * 2E-6
t = np.arange(0, TMAX + 1) * iter2ns   # The time in nanoseconds
dt = np.diff(t)[0]

t0 = time()
fluc_corr_func, fluc_timescales, fluc_expotime = tools.summarize_correlation(fluc_data, ap_indices, dt, TMAX, CUTOFF, SKIP, MAXFRAME)
fixed_corr_func, fixed_timescales, fixed_expotime = tools.summarize_correlation(fixed_data, ap_indices, dt, TMAX, CUTOFF, SKIP, MAXFRAME)
print('Autocorrelation analysis took {0:.1f} minutes'.format((time() - t0) / 60.0))

# Turn autocorrelation results to numpy arrays for easy analysis:
#fluc_corr_func = np.array(fluc_corr_func)
#fluc_timescales = np.array(fluc_timescales)
#fixed_corr_func = np.array(fixed_corr_func)
#fixed_timescales = np.array(fixed_timescales)

# Get the effective interaction time
print('\nAutocorrelation time for osmostat (ns) = ', np.mean(fluc_timescales), '+/', 2.0*np.std(fluc_timescales)/np.sqrt(len(fluc_timescales)))
print('Exponential decay time for osmostat (ns) = ', np.mean(fluc_expotime), '+/', 2.0*np.std(fluc_expotime)/np.sqrt(len(fluc_expotime)))

print('\nAutocorrelation time for fixed-fraction (ns) = ', np.mean(fixed_timescales), '+/', 2.0*np.std(fixed_timescales)/np.sqrt(len(fixed_timescales)))
print('Exponential decay time for fixed-fraction (ns) = ', np.mean(fixed_expotime), '+/', 2.0*np.std(fixed_expotime)/np.sqrt(len(fixed_expotime)))

# Bootstrap confidence estimates and confidence intervals
NBOOTS = 10000
fluc_corr_samps, fluc_time_samps, fluc_expotime_samps = tools.bootstrap_correlation_functions(fluc_corr_func, dt, NBOOTS)
fixed_corr_samps, fixed_time_samps, fixed_expotime_samps = tools.bootstrap_correlation_functions(fixed_corr_func, dt, NBOOTS)
print('\nBootstrap autocorrelation time osmostat (ns) = ', np.mean(fluc_time_samps), '[', np.percentile(fluc_time_samps, 2.5), 'to', np.percentile(fluc_time_samps, 97.5),']')
print('Bootstrap exponential decay time osmostat (ns) = ', np.mean(fluc_expotime_samps), '[', np.percentile(fluc_expotime_samps, 2.5), 'to', np.percentile(fluc_expotime_samps, 97.5),']')

print('\nBootstrap fixed fraction (ns) = ', np.mean(fixed_time_samps), '[', np.percentile(fixed_time_samps, 2.5), 'to', np.percentile(fixed_time_samps, 97.5),']')
print('Bootstrap exponential decay time fixed fraction (ns) = ', np.mean(fixed_expotime_samps), '[', np.percentile(fixed_expotime_samps, 2.5), 'to', np.percentile(fixed_expotime_samps, 97.5),']')


# Extract the acceptance probability for insertion and deletion moves. This is important for accounting for the
# increased amount of time saltswap takes.
naccepted = 0
nattempted = 0
for i in range(1,4):
    ncfile = Dataset("../testsystems/dna_dodecamer/200mM/out{0}.nc".format(i), 'r')
    naccepted += ncfile.groups['Sample state data']['naccepted'][:,0][-1]
    nattempted += ncfile.groups['Sample state data']['nattempted'][:,0][-1]
    ncfile.close()
acc_prob = naccepted / nattempted
print('\nAcceptance probability for saltswap =', acc_prob)

ncmc_steps = 10000   # The number of perturbation steps (1000) multiplied by the number propagation steps (10).
md_steps = 2000
ncmc_eff = (ncmc_steps * acc_prob + md_steps) / md_steps
ncmc_total = (ncmc_steps + md_steps) / md_steps
print('Effective sampling ratio with accepted NCMC attempts', ncmc_eff)
print('Walltime sampling ratio with all NCMC attempts', ncmc_total)

# Plotting the results
FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 11
LINEWIDTH = 3
DPI = 300
FIGSIZE = (6, 5)  # Figure dimension in inches
iter2ns = 2000 * SKIP * 2E-6
FLUC_COL = 'C4'
FIXED_COL = 'C2'

XLIM = (0, 3)
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.15, right=0.95)

# Plot the autocorrelation of the osmostated MD
median = np.percentile(fluc_corr_samps, 50.0, axis=0)
lower = np.percentile(fluc_corr_samps, 2.5, axis=0)
upper = np.percentile(fluc_corr_samps, 97.5, axis=0)
ax.plot(t, median, color=FLUC_COL, lw=LINEWIDTH, label='200mM osmostat')
#ax.plot(t, np.mean(fluc_corr_samps, axis=0), color=FLUC_COL, lw=LINEWIDTH, label='200mM osmostat')
ax.fill_between(t, lower, upper, alpha=0.2, lw=0, color=FLUC_COL)
# Account for effective number of samples with accepted NCMC moves:
ax.plot(t*ncmc_eff, median, color=FLUC_COL, lw=LINEWIDTH-1, label='200mM osmostat, effective NCMC samples', ls='--')
# Account for the total number of MD steps included all NCMC steps
ax.plot(t*ncmc_total, median, color=FLUC_COL, lw=LINEWIDTH, label='200mM osmostat, all NCMC steps', ls=':')

# Plot the autocorrelation of the normal MD cation-phosphate interaction.
median = np.percentile(fixed_corr_samps, 50.0, axis=0)
lower = np.percentile(fixed_corr_samps, 2.5, axis=0)
upper = np.percentile(fixed_corr_samps, 97.5, axis=0)
#plt.plot(t, fixed_corr_func.mean(axis=0), color=FIXED_COL, lw=LINEWIDTH, label='200mM fixed number fraction')
ax.plot(t, median, color=FIXED_COL, lw=LINEWIDTH, label='200mM fixed number fraction')
ax.fill_between(t, lower, upper, alpha=0.2, lw=0, color=FIXED_COL)

ax.grid(ls='--')
ax.legend(fontsize=LEGENDSIZE)
ax.set_xlabel('Time (ns)', fontsize=FONTSIZE)
ax.set_ylabel("Autocorrelation", fontsize=FONTSIZE)
ax.set_xlim(XLIM)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('dna_autocorrelation_5_angs.png', dpi=DPI)
