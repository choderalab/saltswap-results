import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import ncmc_analysis_tools as tools


# Timeseries parameters
DISCARD = 10
FAST = True

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 12
MARKERSIZE = 7
DPI = 300
FIGSIZE = (6, 5)  # Figure dimension in inches
NPROP = 10  # The number of propagation steps per perturbation step
# The colors for the different water models
t4p_col = 'C2'
t3p_col = 'C4'

# Extract the NCMC analysis data and analyze
t3p_folders = glob('../ncmc_optimization/tip3p/nprop{0}/npert_*'.format(NPROP))
t3p = tools.AutoAnalyzeNCMCOptimization(t3p_folders)
t4p_folders = glob('../ncmc_optimization/tip4pew/nprop{0}/npert_*'.format(NPROP))
t4p = tools.AutoAnalyzeNCMCOptimization(t4p_folders)

# Extract the instantaneous acceptance probabilities:
t3_inst_folder = glob('../ncmc_optimization/tip3p/nprop1/npert_1')
t3_inst = tools.AutoAnalyzeNCMCOptimization(t3_inst_folder, nprop=1)
#print('Mean instantaneous TIP3P log-acceptance probability = {0:.2f} +/- {1:.2f}'.format(t3_inst.log_accept[0], t3_inst.log_accept_error[0]*2))
print('Mean instantaneous TIP3P acceptance probability = {0} [{1}, {2}]'.format(t3_inst.booted_accept[0][0], t3_inst.booted_accept[0][1], t3_inst.booted_accept[0][2]))

# Extract the instantaneous acceptance probabilities:
t4_inst_folder = glob('../ncmc_optimization/tip4pew/nprop1/npert_1')
t4_inst = tools.AutoAnalyzeNCMCOptimization(t4_inst_folder, nprop=1)
#print('Mean instantaneous TIP4P-Ew log-acceptance probability = {0:.2f} +/- {1:.2f}'.format(t4_inst.log_accept[0], t4_inst.log_accept_error[0]*2))
print('Mean instantaneous TIP4P-Ew acceptance probability = {0} [{1}, {2}]'.format(t4_inst.booted_accept[0][0], t4_inst.booted_accept[0][1], t4_inst.booted_accept[0][2]))

#----------- FIGURE 1---------------#
# The acceptence probability

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.11)
ax = fig.add_subplot(111)

ax.errorbar(t3p.protocol_length * NPROP, t3p.accept, yerr=t3p.accept_error*2, fmt='o', label='TIP3P', color=t3p_col, markersize=MARKERSIZE)
ax.errorbar(t4p.protocol_length * NPROP, t4p.accept, yerr=t4p.accept_error*2, fmt='o', label='TIP4P-Ew', color=t4p_col, markersize=MARKERSIZE)

#ax.plot(t3p.protocol_length, t3p.accept, color=t3p_col)
#ax.plot(t4p.protocol_length, t4p.accept, color=t4p_col)

plt.xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
plt.ylabel('Acceptance probability $A$', fontsize=FONTSIZE)
plt.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)
ax.grid(ls='--')
plt.savefig('ncmc_acceptance.png', dpi=DPI)
#----------- FIGURE 2 ----------------#
# The average time for a single iteration (MD + insertion deletion attempt) in tip3p is given by
T = 0.006407 # (in seconds) 95% uncertainty = +/- 7.97073894392326e-05 seconds.

# The efficiency of instantaneous insertions in tip3p is:
t3_inst_eff = t3_inst.booted_accept[0][0] / T

# The calculated efficiencies for NCMC protocols. These will be plotted relative to instantaneous insertions.s
t3p_efficieny, t3p_efficiency_error = t3p.calc_efficiency(mode='acceptance')
t4p_efficieny, t4p_efficiency_error = t4p.calc_efficiency(mode='acceptance')

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.11)

ax = fig.add_subplot(111)

graph_nudge = 0.1

ax.errorbar(t3p.protocol_length * NPROP, 1E-46 * t3p_efficieny/t3_inst_eff, yerr=1E-46 *t3p_efficiency_error*2/t3_inst_eff, fmt='o', label='TIP3P', color=t3p_col, markersize=MARKERSIZE)
ax.errorbar(t4p.protocol_length * NPROP, 1E-46 * t4p_efficieny/t3_inst_eff, yerr=1E-46 *t4p_efficiency_error*2/t3_inst_eff, fmt='o', label='TIP4Pew', color=t4p_col, markersize=MARKERSIZE)
ax.set_xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
ax.set_ylabel('Relative efficiency $E$' +r'  ($1 \times 10^{46}$)', fontsize=FONTSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)
ax.grid(ls='--')

plt.savefig('ncmc_efficiency.png', dpi=DPI)
