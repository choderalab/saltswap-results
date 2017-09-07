import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import misc_tools as tools

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 13
LEGENDSIZE = 12
LINEWIDTH = 3
DPI = 300

def get_salt_numbers(file, target_conc):
    """
    Return approximations to the number of salt molecules that would be added to a system in fixed salt-fraction
    simulations.

    Parameters
    ----------
    file: str
        the name of the netcdf file that contains the simulation data.
    target_conc: float
        the required concentration of salt in M (not mM).

    Returns
    -------
    approx_num: float
        the concentration of salt in mM (not M) that would be achieved if salt molecules are added to the system using a
        a typical protocol for fixed-salt simulations. The protocol is based on the openmm.modeller protocol.
    approx_vol: float
        the concentration of salt in mM (not M) that would be achieved if salt molecules are added based on the average
        volume of the system.

    """
    water_conc = 55.4

    ncfile = Dataset(file, 'r')
    volume = ncfile.groups['Sample state data']['volume'][:]
    nspecies = ncfile.groups['Sample state data']['species counts'][:]
    ncfile.close()

    mean_vol = np.mean(volume)

    # The number of salt by the total (mean) volume
    nsalt_guess = int(np.floor(target_conc * mean_vol / 1.66054))
    approx_vol = 1.66054 * nsalt_guess / mean_vol

    # The number of salt by total number of waters - number of counterions
    nmutable_residues = nspecies[0, 0] + np.min(nspecies[0, 1:3]) * 2
    approx_num = int(np.floor(nmutable_residues * target_conc / water_conc)) / mean_vol * 1.66054

    return approx_num*1000, approx_vol*1000


SALCONC = 0.200

conc_spread = np.linspace(0.0, 400.0)
bw = 0.25
XLIM = (0.0, 400.)
xstep = 100
XTICKS = np.arange(XLIM[0], XLIM[1] + xstep, step=xstep)
salt_col = 'C4'
istrgth_col = 'C1'
apprx_num_col = 'C2'
apprx_vol_col = 'C3'

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Water
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/waterbox/200mM/60Angs/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

kernel = gaussian_kde(np.hstack([*salt_conc]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 0].plot(conc_spread, density, lw=LINEWIDTH, color=salt_col, label='Salt concentration')

kernel = gaussian_kde(np.hstack([*ionic_strength]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 0].plot(conc_spread, density, lw=LINEWIDTH, color=istrgth_col, label='Ionic strength', ls='--')

approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)

#ax[0, 0].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)
ax[0, 0].axvline(approx_num, ls='-', color=apprx_num_col, lw=LINEWIDTH, label='Fixed salt')
# ax[0,0].axvline(approx_vol, ls=':', color=apprx_vol_col, lw=LINEWIDTH)
ax[0, 0].legend(loc=2, fontsize=FONTSIZE)
ax[0, 0].set_xlim(XLIM)
ax[0, 0].set_title('TIP3P water ({0} waters)'.format(ntotal), fontsize=FONTSIZE - 1)
ax[0, 0].set_ylabel('Relative density', fontsize=FONTSIZE)

# DHFR
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/dhfr/200mM/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

kernel = gaussian_kde(salt_conc.flatten(), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 1].plot(conc_spread, density, lw=LINEWIDTH, color=salt_col)

kernel = gaussian_kde(np.hstack([*ionic_strength]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 1].plot(conc_spread, density, lw=LINEWIDTH, color=istrgth_col, label='Ionic strength', ls='--')

approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)
ax[0, 1].axvline(approx_num, ls='-', color=apprx_num_col, lw=LINEWIDTH)
# ax[0,1].axvline(approx_vol, ls=':', color=apprx_vol_col, lw=LINEWIDTH)
#ax[0, 1].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[0, 1].set_xlim(XLIM)
ax[0, 1].set_title('DHFR ({0} waters)'.format(ntotal), fontsize=FONTSIZE - 1)

# SRC
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/src/200mM/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

kernel = gaussian_kde(np.hstack([*salt_conc]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[1, 0].plot(conc_spread, density, lw=LINEWIDTH, color=salt_col)

kernel = gaussian_kde(np.hstack([*ionic_strength]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[1, 0].plot(conc_spread, density, lw=LINEWIDTH, color=istrgth_col, label='Ionic strength', ls='--')

approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)
ax[1, 0].axvline(approx_num, ls='-', color=apprx_num_col, lw=LINEWIDTH)
# ax[1,0].axvline(approx_vol, ls=':', color=apprx_vol_col, lw=LINEWIDTH)
#ax[1, 0].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[1, 0].set_xlim(XLIM)
ax[1, 0].set_title('Src kinase ({0} waters)'.format(ntotal), fontsize=FONTSIZE - 1)
ax[1, 0].set_ylabel('Relative density', fontsize=FONTSIZE)
ax[1, 0].set_xlabel('Concentration (mM)', fontsize=FONTSIZE)

# DNA
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../testsystems/dna_dodecamer/200mM/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc, nsalt, ntotal = tools.read_species_concentration(files)

kernel = gaussian_kde(np.hstack(*[salt_conc]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[1, 1].plot(conc_spread, density, lw=LINEWIDTH, color=salt_col)

kernel = gaussian_kde(np.hstack([*ionic_strength]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[1, 1].plot(conc_spread, density, lw=LINEWIDTH, color=istrgth_col, label='Ionic strength', ls='--')

approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)
ax[1, 1].axvline(approx_num, ls='-', color=apprx_num_col, lw=LINEWIDTH)
# ax[1,1].axvline(approx_vol, ls=':', color=apprx_vol_col, lw=LINEWIDTH)
#ax[1, 1].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[1, 1].set_xlim(XLIM)
ax[1, 1].set_title('DNA dodecamer ({0} waters)'.format(ntotal), fontsize=FONTSIZE - 1)
ax[1, 1].set_xlabel('Concentration (mM)', fontsize=FONTSIZE)

for i in (0, 1):
    for j in (0, 1):
        for label in (ax[i, j].get_xticklabels() + ax[i, j].get_yticklabels()):
            label.set_fontsize(TICKSIZE)

for i in (0, 1):
    for j in (0, 1):
        ax[i, j].set_xticks(XTICKS)

plt.tight_layout()
plt.savefig('salt_densities.png', dpi=DPI)
