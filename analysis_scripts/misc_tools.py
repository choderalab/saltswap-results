import numpy as np
from netCDF4 import Dataset

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
        the salt concentration in mM.
    ionic_strength:
        the molar ionic strength in mM.
    cation_conc: numpy.ndarray
        thr cation concentration in mM.
    anion_conc: numpy.ndarray
        the anion concentration in mM.
    nsalt: numpy.ndarray
        the number of neutral salt pairs in the system.
    ntotal_species: int
        the total number of water molecules, cations and anions.
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
        ntotal_species = nspecies[0,:].sum()
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


    return 1000*np.array(salt_conc), 1000*np.array(ionic_strength), 1000*np.array(cation_conc), 1000*np.array(anion_conc), nsalt, ntotal_species
