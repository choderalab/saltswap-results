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

# Extract the NCMC analysis data and analyze
n1_folders = glob('../tip3p/npert_*')
n1 = tools.AutoAnalyzeNCMCOptimization(n1_folders, nprop=1)

n5_folders = glob('../npert_nprop/nprop5/npert_*')
n5 = tools.AutoAnalyzeNCMCOptimization(n5_folders, nprop=5)

n10_folders = glob('../npert_nprop/nprop10/npert_*')
n10 = tools.AutoAnalyzeNCMCOptimization(n10_folders, nprop=10)

n20_folders = glob('../npert_nprop/nprop20/npert_*')
n20 = tools.AutoAnalyzeNCMCOptimization(n20_folders, nprop=20)

nprops = [n1, n5, n10, n20]
leg_names = ['1', '5', '10', '20']
#----------- FIGURE 1---------------#
# The acceptence probability

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

for i in range(len(nprops)):
    plt.errorbar(nprops[i].protocol_length, nprops[i].accept, yerr=nprops[i].accept_error*2, fmt='o', label=leg_names[i])

plt.xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
plt.ylabel('Acceptance Probability', fontsize=FONTSIZE)
plt.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('ncmc_propagation_acceptance_tip3p.png', dpi=DPI)

#----------- FIGURE 2 ----------------#

# Calculate efficiency
eff = []
err = []
for n in nprops:
    efficieny, error = n.calc_efficiency(mode='acceptance')
    eff.append(efficieny)
    err.append(error)
max_eff = np.max(np.hstack([*eff]))

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

graph_nudge = 0.1

for i in range(len(nprops)):
    ax.errorbar(nprops[i].protocol_length, eff[i] / max_eff, yerr=err[i] * 2 / max_eff, fmt='o',
                label=leg_names[i])

ax.set_ylim((0 - graph_nudge, 1.0 + graph_nudge))
ax.set_xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
ax.set_ylabel('Relative efficiency', fontsize=FONTSIZE)
ax.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('ncmc_propagation_efficiency_tip3p.png', dpi=DPI)
