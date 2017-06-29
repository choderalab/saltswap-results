import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import analsys_tools as tools


# Timeseries parameters
DISCARD = 10
FAST = True

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 12
DPI = 300
FIGSIZE = (5, 5)  # Figure dimension in inches
# The colors for the different water models
t4p_col = 'C0'
t3p_col = 'C1'

# Extract the NCMC analysis data and analyze
t3p_folders = glob('../tip3p/npert_*')
t3p = tools.AutoAnalyzeNCMCOptimization(t3p_folders)
t4p_folders = glob('../tip4pew/npert_*')
t4p = tools.AutoAnalyzeNCMCOptimization(t4p_folders)

#----------- FIGURE 1---------------#
# The acceptence probability

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

ax.errorbar(t3p.protocol_length, t3p.accept, yerr=t3p.accept_error*2, fmt='o', label='TIP3P', color=t3p_col)
ax.errorbar(t4p.protocol_length, t4p.accept, yerr=t4p.accept_error*2, fmt='o', label='TIP4Pew', color=t4p_col)

#ax.plot(t3p.protocol_length, t3p.accept, color=t3p_col)
#ax.plot(t4p.protocol_length, t4p.accept, color=t4p_col)

plt.xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
plt.ylabel('Acceptance Probability', fontsize=FONTSIZE)
plt.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('ncmc_acceptance.png', dpi=300)

#----------- FIGURE 2 ----------------#
t3p_efficieny, t3p_efficiency_error = t3p.calc_efficiency(mode='acceptance')
t4p_efficieny, t4p_efficiency_error = t4p.calc_efficiency(mode='acceptance')

max_eff = np.max(np.hstack((t3p_efficieny,t4p_efficieny)))

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

graph_nudge = 0.1

ax.errorbar(t3p.protocol_length, t3p_efficieny/max_eff, yerr=t3p_efficiency_error*2/max_eff, fmt='o', label='TIP3P', color=t3p_col)
ax.errorbar(t4p.protocol_length, t4p_efficieny/max_eff, yerr=t4p_efficiency_error*2/max_eff, fmt='o', label='TIP4Pew', color=t4p_col)
ax.set_ylim((0 - graph_nudge, 1.0 + graph_nudge))
ax.set_xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
ax.set_ylabel('Relative efficiency', fontsize=FONTSIZE)
ax.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('ncmc_efficiency.png', dpi=300)
