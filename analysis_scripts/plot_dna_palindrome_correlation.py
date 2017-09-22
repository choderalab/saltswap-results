import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import dna_analysis_tools as tools
import mdtraj

# Loading the simulations
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

# Analyze the correlation of the cation occupancies of the dna palindrome
SKIP = 10
iter2ns = 2000 * SKIP * 2E-6 # The conversion between iteration and nanoseconds of MD
MAXFRAME = None    # Analysing up to 40 ns.


cor_fluc = []
cor_fixed = []
for i in range(3):
    cor_fluc.append(tools.mirror_occupancy_correlation(fluc_data[i][0], fluc_data[i][1], skip=SKIP, maxframe=MAXFRAME))
    cor_fixed.append(tools.mirror_occupancy_correlation(fixed_data[i][0], fixed_data[i][1], skip=SKIP, maxframe=MAXFRAME))

# Some of the saltswap simulations didn't complete, so taking the shortest length as the maximum iteration for all plotting.
max_iter = np.inf
for cor in cor_fluc:
    if cor.shape[0] < max_iter:
        max_iter = cor.shape[0]

cor_fixed_arr = np.array(cor_fixed)[:,0:max_iter]
cor_fluc_arr = np.array([c[0:max_iter] for c in cor_fluc])
t_fluc = np.arange(1, max_iter + 1) * iter2ns   # The time in nanoseconds.

# The actual plotting
FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 11
LINEWIDTH = 3
DPI = 300
FIGSIZE = (6, 5)  # Figure dimension in inches
FLUC_COL = 'C4'
FIXED_COL = 'C2'

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111)

XLIM = (0,45)
ax.plot(t_fluc, cor_fluc_arr.mean(axis=0), color=FLUC_COL, label='200mM osmostat', linewidth=LINEWIDTH)
ax.fill_between(t_fluc, cor_fluc_arr.min(axis=0), cor_fluc_arr.max(axis=0), alpha=0.2, color=FLUC_COL, lw=0)
ax.plot(t_fluc, cor_fixed_arr.mean(axis=0), color=FIXED_COL, label='200mM fixed number fraction', linewidth=LINEWIDTH)
ax.fill_between(t_fluc, cor_fixed_arr.min(axis=0), cor_fixed_arr.max(axis=0), alpha=0.2, color=FIXED_COL, lw=0)

ax.legend(loc=4, fontsize=LEGENDSIZE)
ax.set_xlim(XLIM)
ax.set_xlabel('Time ($ns$)', fontsize=FONTSIZE)
ax.set_ylabel("Correlation coefficient", fontsize=FONTSIZE)
ax.grid(ls='--')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('dna_palindrome_correlation.png', dpi=DPI)
