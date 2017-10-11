import numpy as np
import matplotlib.pyplot as plt
import misc_tools
from scipy.stats import gaussian_kde
import dna_analysis_tools as tools

FONTSIZE = 15
TICKSIZE = 12
LEGENDSIZE = 11
LINEWIDTH = 3
DPI = 300
FIGSIZE = (6, 5)  # Figure dimension in inches

osmo_col = 'C4'
osmo_big_col = 'C6'
fixed_col = 'C2'
fixed_big_col = 'C8'

#------ Helper functions for plotting -------#
def plot_dna_charge_rdfs(ax_element, rdfs, bins, label, color, linewidth=2):
    """
    Wrapper for plotting charge radial distribution.
    """
    mean_rdf_fluc = np.mean(rdfs, axis=0)
    ax_element.plot(bins, mean_rdf_fluc, color=color, label=label, linewidth=linewidth)
    ax_element.fill_between(bins, np.percentile(rdfs, 2.5, axis=0), np.percentile(rdfs, 97.5, axis=0), alpha=0.3, color=color)


def plot_density(ax_element, files, conc_spread, color, bw=0.25, alpha=0.3, linewidth=2):
    """
    Wrapper for plotting a kernel estimate of salt density.
    """
    salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = misc_tools.read_species_concentration(files)

    kernel = gaussian_kde(np.hstack(salt_conc), bw)
    density = kernel(conc_spread)
    density = density / np.max(density)
    ax_element.plot(conc_spread, density, lw=linewidth, color=color)
    ax_element.fill_between(conc_spread, 0.0, density, color=color, alpha=alpha)
    ax_element.set_yticks([])

# The same amount of simulation time will be discarded from all simulations, and data will be analysed at the following
# consistent steps in time:
equil_time = 10.0 # nanoseconds
skip_time = 1.0 # nanoseconds

bins = np.linspace(1, 30)   # The distances where the charge RDF will be estimated, in Angstroms.

#-------- Loading the simulations from the smaller box of water -----------#
# Accounting for the frequency with which the data was saved:
save_freq = 1
iter2ns = 2000 * save_freq * 2E-6      # The conversion between the simulation iteration in each traj object to nanoseconds.

# Discarding the 1st 10ns as equilibration.
min_frame = int(np.floor(equil_time / iter2ns))

# Measuring the charge distribition at least every 1 ns.
skip = int(np.ceil(skip_time / iter2ns))

# The repeat numbering scheme:
repeats = [1, 2, 3]

# Osmostat simulations
folder = '../testsystems/dna_dodecamer/200mM'
rdfs_osmostat = tools.wrapper_dna_charge_rdf(folder, repeats, skip=skip, min_frame=min_frame, bins=bins, nboots=1000)

# Fixed salt simulations
folder = '../testsystems/dna_dodecamer/200mM_fixed_number/'
rdfs_fixed = tools.wrapper_dna_charge_rdf(folder, repeats, skip=skip, min_frame=min_frame, bins=bins, nboots=1000)

#-------- Loading the simulations from the larger box of water -----------#
# Accounting for the frequency with which the data was saved:
save_freq = 4
iter2ns = 2000 * save_freq * 2E-6      # The conversion between the simulation iteration in each traj object to nanoseconds.

# Discarding the 1st 10ns as equilibration.
min_frame = int(np.floor(equil_time / iter2ns))

# Measuring the charge distribition at least every 1 ns.
skip = int(np.ceil(skip_time / iter2ns))

repeats = [1, 2, 3]

# Osmostat simulations
folder = '../testsystems/dna_dodecamer/solvent_padding_16/200mM'
rdfs_osmostat_big = tools.wrapper_dna_charge_rdf(folder, repeats, skip=skip, min_frame=min_frame, bins=bins, nboots=1000)

# Fixed salt simulations
folder = '../testsystems/dna_dodecamer/solvent_padding_16/200mM_fixed_number'
rdfs_fixed_big = tools.wrapper_dna_charge_rdf(folder, repeats, skip=skip, min_frame=min_frame, bins=bins, nboots=1000)

#------ Quick analysis------#
# Quantifying the difference between the charge distributions of the fixed fraction and osmostat simulations:
diffs = (rdfs_osmostat - rdfs_fixed)
mean_diffs = np.mean(diffs,axis=1)
lower = np.percentile(mean_diffs, 2.5)
upper =  np.percentile(mean_diffs, 97.5)
print('4296 waters: Mean difference between osmostat and fixed fraction charge distribution = {0:.2f} [{1:.2f}, {2:.2f}]'.format(np.mean(mean_diffs), lower, upper))

diffs = (rdfs_osmostat_big - rdfs_fixed_big)
mean_diffs = np.mean(diffs,axis=1)
lower = np.percentile(mean_diffs, 2.5)
upper =  np.percentile(mean_diffs, 97.5)
print('9276 waters: Mean difference between osmostat and fixed fraction charge distribution = {0:.2f} [{1:.2f}, {2:.2f}]'.format(np.mean(mean_diffs), lower, upper))

#-------- The actual plotting -----------#
fig, ax1 = plt.subplots(figsize=FIGSIZE)

# The main plot:
plot_dna_charge_rdfs(ax1, rdfs_osmostat, bins, 'Osmostat (4296 waters)', osmo_col, linewidth=LINEWIDTH)
plot_dna_charge_rdfs(ax1, rdfs_osmostat_big, bins, 'Osmostat (9276 waters)', osmo_big_col, linewidth=LINEWIDTH)
plot_dna_charge_rdfs(ax1, rdfs_fixed, bins, 'Fixed (4296 waters)', fixed_col, linewidth=LINEWIDTH)
plot_dna_charge_rdfs(ax1, rdfs_fixed_big, bins, 'Fixed (9276 waters)', fixed_big_col, linewidth=LINEWIDTH)

ax1.grid(ls='--')
ax1.set_xlabel('Distance from DNA (Å)', fontsize=FONTSIZE)
ax1.set_ylabel('Total charge ($e$)', fontsize=FONTSIZE)
ax1.legend(loc='lower left', fontsize=LEGENDSIZE)
ax1.set_xlim((1,30))
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

# The inset plot:
left, bottom, width, height = [0.58, 0.26, 0.3, 0.35]
ax2 = fig.add_axes([left, bottom, width, height])

conc_spread = np.linspace(0.0, 300.0)

files = ['../testsystems/dna_dodecamer/200mM/out{0}.nc'.format(r) for r in [1,2,3]]
plot_density(ax2, files, conc_spread, osmo_col, linewidth=LINEWIDTH - 1)
files = ['../testsystems/dna_dodecamer/solvent_padding_16/200mM/out{0}.nc'.format(r) for r in [1,2,3]]
plot_density(ax2, files, conc_spread, osmo_big_col, linewidth=LINEWIDTH - 1)
ax2.axvline(200, color='black', ls='--')

ax2.set_xlabel('Concentration $c$ (mM)', fontsize=FONTSIZE-3)
for label in (ax2.get_xticklabels()):
    label.set_fontsize(TICKSIZE)

plt.savefig('dna_size_dependence.png', dpi=DPI)
