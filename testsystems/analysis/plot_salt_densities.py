import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Plotting parameters
FONTSIZE = 15
TICKSIZE = 13
LEGENDSIZE = 12
LINEWIDTH = 3
DPI = 300


def read_species_concentration(files):
    """
    Extract ion concentration from a number of simulation data files.

    Parameters
    ----------
    files: list of str
        The netcdf files whose salt concentrations will be calculated.

    Returns
    -------
    salt_conc: numpy.ndarray
        the salt concentration in M.
    ionic_strength:
        the molar ionic strength in M.
    cation_conc: numpy.ndarray
        thr cation concentration in M.
    anion_conc: numpy.ndarray
        the anion concentration in M.
    """
    salt_conc = []
    anion_conc = []
    cation_conc = []
    ionic_strength = []

    for file in files:
        ncfile = Dataset(file, 'r')
        volume = ncfile.groups['Sample state data']['volume'][:]
        nspecies = ncfile.groups['Sample state data']['species counts'][:]
        ncfile.close()

        # Record the salt concentration as the number of neutralizing ions.
        nsalt = np.min(nspecies[:, 1:3], axis=1)
        salt_conc.append(1.0 * nsalt / volume * 1.66054)

        # Recording charge and concentration of the biomolecule
        ncation = nspecies[:, 1]
        nanion = nspecies[:, 2]
        biomol_charge = nanion - ncation
        bc = 1. / volume * 1.66054

        # Record the concentration of each species seperately
        cc = 1.0 * ncation / volume * 1.66054  # cation concentration in M
        ac = 1.0 * nanion / volume * 1.66054  # anion concentration in M
        cation_conc.append(cc)
        anion_conc.append(ac)

        # Record the ionic strength.
        ionic_strength.append((ac + cc) / 2.0)
        # If one wants to include the ionic strength of the biomolecule:
        # ionic_strength.append((ac + cc + bc*(biomol_charge**2))/2.0)

    return np.array(salt_conc), np.array(ionic_strength), np.array(cation_conc), np.array(anion_conc)


def get_salt_numbers(file, target_conc):
    """
    Return approximations to the number of salt molecules that should be added to a system.
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

    return approx_num, approx_vol


SALCONC = 0.200

conc_spread = np.linspace(0.0, 0.4)
bw = 0.25
XLIM = (0.0, 0.4)
xstep = 0.1
XTICKS = np.arange(XLIM[0], XLIM[1] + xstep, step=xstep)
salt_col = 'C4'
istrgth_col = 'C1'
apprx_num_col = 'C2'
apprx_vol_col = 'C3'

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Water
files = ['../../calibration/equilibrium_staging/tip3p/deltamu_317.61/out.nc']
salt_conc, ionic_strength, cation_conc, anion_conc = read_species_concentration(files)

kernel = gaussian_kde(np.hstack([*salt_conc]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 0].plot(conc_spread, density, lw=LINEWIDTH, color=salt_col, label='Salt')

kernel = gaussian_kde(np.hstack([*ionic_strength]), bw)
density = kernel(conc_spread)
density = density / np.max(density)
ax[0, 0].plot(conc_spread, density, lw=LINEWIDTH, color=istrgth_col, label='Ionic strength', ls='--')

approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)

ax[0, 0].legend(loc=2, fontsize=FONTSIZE)
ax[0, 0].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
approx_num, approx_vol = get_salt_numbers(files[0], SALCONC)
ax[0, 0].axvline(approx_num, ls='-', color=apprx_num_col, lw=LINEWIDTH)
# ax[0,0].axvline(approx_vol, ls=':', color=apprx_vol_col, lw=LINEWIDTH)
ax[0, 0].set_xlim(XLIM)
ax[0, 0].set_title('Box of 4085 water molecules', fontsize=FONTSIZE - 1)
ax[0, 0].set_ylabel('Relative density', fontsize=FONTSIZE)

# DHFR
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../dhfr/200mM/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc = read_species_concentration(files)

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
ax[0, 1].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[0, 1].set_xlim(XLIM)
ax[0, 1].set_title('DHFR with 7023 water molecules', fontsize=FONTSIZE - 1)

# SRC
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../src/200mM/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc = read_species_concentration(files)

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
ax[1, 0].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[1, 0].set_xlim(XLIM)
ax[1, 0].set_title('Src kinase with 17398 water molecules', fontsize=FONTSIZE - 1)
ax[1, 0].set_ylabel('Relative density', fontsize=FONTSIZE)
ax[1, 0].set_xlabel('Concentration (M)', fontsize=FONTSIZE)

# DNA
file_names = ['out1.nc', 'out2.nc', 'out3.nc']
files = ['../dna_dodecamer/' + f for f in file_names]
salt_conc, ionic_strength, cation_conc, anion_conc = read_species_concentration(files)

kernel = gaussian_kde(salt_conc.flatten(), bw)
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
ax[1, 1].axvline(SALCONC, ls='-', color='grey', label='Macroscopic concentration', lw=2, alpha=0.5)
ax[1, 1].set_xlim(XLIM)
ax[1, 1].set_title('DNA palindrome with 4296 water molecules', fontsize=FONTSIZE - 1)
ax[1, 1].set_xlabel('Concentration (M)', fontsize=FONTSIZE)

for i in (0, 1):
    for j in (0, 1):
        for label in (ax[i, j].get_xticklabels() + ax[i, j].get_yticklabels()):
            label.set_fontsize(TICKSIZE)

for i in (0, 1):
    for j in (0, 1):
        ax[i, j].set_xticks(XTICKS)

plt.tight_layout()
plt.savefig('salt_densities.png', dpi=DPI)