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
ALPHA = 0.3
FIGSIZE = (6, 5)  # Figure dimension in inches
NBOOTS = 10000      # The number of bootstrap samples
osmo_col = 'C4'
osmo_big_col = 'C6'
fixed_col = 'C2'
fixed_big_col = 'C8'

#------ Helper functions for plotting -------#
from misc_tools import bootstrap_array, plot_booted_stats

#def plot_dna_charge_rdfs(ax_element, rdfs, bins, label, color, linewidth=2):
#    """
#    Wrapper for plotting charge radial distribution.
#    """
#    mean_rdf_fluc = np.mean(rdfs, axis=0)
#    ax_element.plot(bins, mean_rdf_fluc, color=color, label=label, linewidth=linewidth)
#    ax_element.fill_between(bins, np.percentile(rdfs, 2.5, axis=0), np.percentile(rdfs, 97.5, axis=0), ALPHA=0.3, color=color)


def plot_density(ax_element, files, conc_spread, color, bw=0.25, alpha=ALPHA, linewidth=2):
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
equil_time = 15.0 # nanoseconds
skip_time = 1.0 # nanoseconds

dna_charge = -22.0
distances = np.linspace(1, 30)  # The distances within which the ions will be will be counted.

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
s_osmo, c_osmo, a_osmo = tools.wrapper_ion_distance_profile(folder, repeats, distances, skip=skip, min_frame=min_frame)
a_osmo = bootstrap_array(a_osmo, nboots=NBOOTS)
c_osmo = bootstrap_array(c_osmo, nboots=NBOOTS)
charge_osmo = c_osmo - a_osmo + dna_charge
# Fixed salt simulations
folder = '../testsystems/dna_dodecamer/200mM_fixed_number/'
s_fixed, c_fixed, a_fixed = tools.wrapper_ion_distance_profile(folder, repeats, distances, skip=skip, min_frame=min_frame)
a_fixed = bootstrap_array(a_fixed, nboots=NBOOTS)
c_fixed = bootstrap_array(c_fixed, nboots=NBOOTS)
charge_fixed = c_fixed - a_fixed + dna_charge

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
folder = '../testsystems/dna_dodecamer/solvent_padding_16/200mM/'
s_osmo_big, c_osmo_big, a_osmo_big = tools.wrapper_ion_distance_profile(folder, repeats, distances, skip=skip, min_frame=min_frame)
a_osmo_big = bootstrap_array(a_osmo_big, nboots=NBOOTS)
c_osmo_big = bootstrap_array(c_osmo_big, nboots=NBOOTS)
charge_osmo_big = c_osmo_big - a_osmo_big + dna_charge
# Fixed salt simulations
folder = '../testsystems/dna_dodecamer/solvent_padding_16/200mM_fixed_number/'
s_fixed_big, c_fixed_big, a_fixed_big = tools.wrapper_ion_distance_profile(folder, repeats, distances, skip=skip, min_frame=min_frame)
a_fixed_big = bootstrap_array(a_fixed_big, nboots=NBOOTS)
c_fixed_big = bootstrap_array(c_fixed_big, nboots=NBOOTS)
charge_fixed_big = c_fixed_big - a_fixed_big + dna_charge

#------ Quick analysis------#
# Quantifying the difference between the charge distributions of the fixed fraction and osmostat simulations:
charge_diffs = charge_osmo - charge_fixed
mean_diffs = np.mean(charge_diffs, axis=1)
lower = np.percentile(mean_diffs, 2.5)
upper =  np.percentile(mean_diffs, 97.5)
print('\n4296 waters: Mean difference between osmostat and fixed fraction charge distribution = {0:.2f} [{1:.2f}, {2:.2f}]'.format(np.mean(mean_diffs), lower, upper))

charge_diffs = (charge_osmo_big - charge_fixed_big)
mean_diffs = np.mean(charge_diffs, axis=1)
lower = np.percentile(mean_diffs, 2.5)
upper =  np.percentile(mean_diffs, 97.5)
print('\n9276 waters: Mean difference between osmostat and fixed fraction charge distribution = {0:.2f} [{1:.2f}, {2:.2f}]'.format(np.mean(mean_diffs), lower, upper))

charge_diffs = (charge_osmo_big - charge_osmo)
mean_diffs = np.mean(charge_diffs,axis=1)
lower = np.percentile(mean_diffs, 2.5)
upper =  np.percentile(mean_diffs, 97.5)
print('\nMean difference between large osmostated box and the small osmostated box = {0:.2f} [{1:.2f}, {2:.2f}]\n'.format(np.mean(mean_diffs), lower, upper))

#-------- The actual plotting -----------#
fig, ax1 = plt.subplots(figsize=FIGSIZE)

# The main plot:
#  Osmostated simulations
plot_booted_stats(ax1, charge_osmo, distances, 'Osmostat (4296 waters)', osmo_col, linewidth=2)
boot_samps = bootstrap_array(charge_osmo_big)
plot_booted_stats(ax1, charge_osmo_big, distances, 'Osmostat (9276 waters)', osmo_big_col, linewidth=2)
# Fixed salt simulations
plot_booted_stats(ax1, charge_fixed, distances, 'Fixed (4296 waters)', fixed_col, linewidth=2)
plot_booted_stats(ax1, charge_fixed_big, distances, 'Fixed (9276 waters)', fixed_big_col, linewidth=2)

ax1.grid(ls='--')
ax1.set_xlabel('Distance from DNA (Å)', fontsize=FONTSIZE)
ax1.set_ylabel('Total charge ($e$)', fontsize=FONTSIZE)
ax1.legend(loc='lower left', fontsize=LEGENDSIZE)
ax1.set_xlim((1,30))
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontsize(TICKSIZE)

# The inset plot:
left, bottom, width, height = [0.58, 0.215, 0.3, 0.35]
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
