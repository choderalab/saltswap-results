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
DPI = 300
FIGSIZE = (6, 5)  # Figure dimension in inches

# Extract the NCMC analysis data and analyze
n1_folders = glob('../ncmc_optimization/tip3p/nprop1/npert_*')
n1 = tools.AutoAnalyzeNCMCOptimization(n1_folders, nprop=1)

n5_folders = glob('../ncmc_optimization/tip3p/nprop5/npert_*')
n5 = tools.AutoAnalyzeNCMCOptimization(n5_folders, nprop=5)

n10_folders = glob('../ncmc_optimization/tip3p/nprop10/npert_*')
n10 = tools.AutoAnalyzeNCMCOptimization(n10_folders, nprop=10)

n20_folders = glob('../ncmc_optimization/tip3p/nprop20/npert_*')
n20 = tools.AutoAnalyzeNCMCOptimization(n20_folders, nprop=20)

nprops = [n1, n5, n10, n20]
leg_names = ['1 steps', '5 steps', '10 steps', '20 steps']

#----------- FIGURE 1---------------#
# The acceptence probability

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

for i in range(len(nprops)):
    ax.errorbar(nprops[i].protocol_length, nprops[i].accept, yerr=nprops[i].accept_error*2, fmt='o', label=leg_names[i])
ax.grid(ls='--')
ax.set_xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
ax.set_ylabel('Acceptance Probability', fontsize=FONTSIZE)
ax.legend(fontsize=LEGENDSIZE)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

#plt.savefig('ncmc_propagation_acceptance_tip3p.png', dpi=DPI)

#----------- FIGURE 2 ----------------#
t3_inst_folder = glob('../ncmc_optimization/tip3p/nprop1/npert_1')
t3_inst = tools.AutoAnalyzeNCMCOptimization(t3_inst_folder, nprop=1)
T = 0.006407 # (in seconds) 95% uncertainty = +/- 7.97073894392326e-05 seconds.

# The efficiency of instantaneous insertions in tip3p is:
t3_inst_eff = t3_inst.booted_accept[0][0] / T


# Calculate efficiency
eff = []
err = []
for n in nprops:
    efficieny, error = n.calc_efficiency(mode='acceptance')
    eff.append(efficieny)
    err.append(error)

fig = plt.figure(figsize=FIGSIZE)
fig.subplots_adjust(left=0.17, right=0.95)
ax = fig.add_subplot(111)

for i in range(len(nprops)):
    if i == 0:
        ax.errorbar(nprops[i].protocol_length[1:],  1E-46 * eff[i][1:] / t3_inst_eff, yerr=1E-46 * err[i][1:] * 2 / t3_inst_eff, fmt='o', label=leg_names[i])
    else:
        ax.errorbar(nprops[i].protocol_length, 1E-46 * eff[i] / t3_inst_eff, yerr=1E-46 * err[i] * 2 / t3_inst_eff, fmt='o', label=leg_names[i])

ax.set_xlabel('Length of NCMC protocol (ps)', fontsize=FONTSIZE)
ax.set_ylabel('Relative efficiency' + r'  ($1 \times 10^{46}$)', fontsize=FONTSIZE)
ax.legend(fontsize=LEGENDSIZE)
ax.grid(ls='--')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('ncmc_propagation_efficiency_tip3p.png', dpi=DPI)
