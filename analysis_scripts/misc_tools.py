import numpy as np
from netCDF4 import Dataset
from  scipy.stats import t
import mdtraj

def read_species_concentration(files, verbose=True):
    """
    Extract ion concentration from a number of simulation data files.

    Parameters
    ----------
    files: list of str
        The netcdf files whose salt concentrations will be calculated.
    verbose: bool
        Whether to print out statistics regarding the salt concentrations.

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
    ionic_strength_biomol = []
    salt_number = []
    for file in files:
        ncfile = Dataset(file, 'r')
        volume = ncfile.groups['Sample state data']['volume'][:]
        nspecies = ncfile.groups['Sample state data']['species counts'][:]
        ncfile.close()

        # Record the salt concentration as the number of neutralizing ions.
        ntotal_species = nspecies[0,:].sum()
        nsalt = np.min(nspecies[:, 1:3], axis=1)
        salt_number.append(nsalt)
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

        # Including the ionic strength of the biomolecule:
        ionic_strength_biomol.append((ac + cc + bc*(biomol_charge**2))/2.0)

    salt_conc = 1000*np.hstack(salt_conc)
    lower = np.percentile(salt_conc, 2.5)
    upper = np.percentile(salt_conc, 97.5)
    print('Mean salt concentration = {0:.1f}, with 95% of all samples between {1:.1f} and {2:.1f}'.format(salt_conc.mean(), lower, upper))

    print('Biomolecular charge = {0}e'.format(biomol_charge[0]))
    ionic_strength = 1000*np.hstack(ionic_strength)
    lower = np.percentile(ionic_strength, 2.5)
    upper = np.percentile(ionic_strength, 97.5)
    print('Mean ionic strength of saline buffer = {0:.1f}, with 95% of all samples between {1:.1f} and {2:.1f}'.format(ionic_strength.mean(), lower, upper))
    ionic_strength_biomol = 1000*np.hstack(ionic_strength_biomol)
    lower = np.percentile(ionic_strength_biomol, 2.5)
    upper = np.percentile(ionic_strength_biomol, 97.5)
    print('Mean ionic strength of whole system = {0:.1f}, with 95% of all samples between {1:.1f} and {2:.1f}'.format(ionic_strength_biomol.mean(), lower, upper))

    return salt_conc, ionic_strength, 1000*np.array(cation_conc), 1000*np.array(anion_conc), np.array(salt_number), ntotal_species

def batch_estimate(x, stat_ineff, percentile=95.0):
    """
    Estimate the mean of a variable from MCMC and the standard error using the method of batch means. Batches have a
    number of samples that are least twice the autocorrelation time.

    Referecence:
    James M. Flegal, Murali Haran and Galin L. Jones, Markov Chain Monte Carlo: Can We
    Trust the Third Significant Figure? Statistical Science 2008, Vol. 23, No. 2, 250-260

    The article is also on the arXiv.

    TODO: Augment with "consistent batch means" as described in the reference above.

    Parameters
    ----------
    x: numpy.ndarray
        Samples from MCMC
    stat_ineff: float
        The statistical inefficiency of x.
    percentile: float
        the percentile of the half width asymptotic confidence interval
    Returns
    -------
    mu: float
        the mean of x
    std_errror: float
        the standard error of mu
    conf_width: float
        the half width of the asymptotic confidence interval. The lower and upper estimates are x - conf_width, x + conf_width
    a: int
        the number of batches used.
    """
    # Given the statistical inefficiency, find the autocorrelation time
    tau = 0.5*(stat_ineff - 1)

    # The number of samples in each batch is twice the autocorrelation time
    b = int(np.ceil(tau * 2))

    # The number of even sized batches is then given by
    a_initial = int(np.floor(len(x) / float(b)))

    # Ensure that no batches will extend further than the length of x:
    a = a_initial
    while b * (a + 1) + a > len(x):
        a -= 1

    # Calculate the mean of each batch
    batch_means = np.zeros(a)
    for k in range(a):
        batch_means[k-1] = np.mean(x[k * b + k : b * (k + 1) + k])

    # The mean squared deviation of each batch from the sample mean is an estimate of the standard error of the mean.
    mu = np.mean(x)
    sigma = np.sqrt(b * np.sum((batch_means - mu)**2) / (a - 1))
    std_error = sigma / np.sqrt(len(x))
    conf_width = t.ppf(percentile/100.0, a - 1) * std_error

    return mu, std_error, a, conf_width


def bootstrap_estimates(arr, nboots=10000):
    """
    Bootstrap estimation of the mean of an array
    """
    arr_mean = np.zeros(nboots)
    for b in range(nboots):
        samps = np.random.choice(len(arr), len(arr))
        arr_boot = arr[samps]
        arr_mean[b] = arr_boot.mean()

    return arr_mean

def bootstrap_array(arr, nboots=10000):
    """
    Bootstrap estimation of the mean of an the columns of an array
    """
    arr_mean = np.zeros((nboots, arr.shape[1]))
    for b in range(nboots):
        samps = np.random.choice(arr.shape[0], arr.shape[0])
        arr_boot = arr[samps, :]
        arr_mean[b, :] = arr_boot.mean(axis=0)
    return arr_mean


def plot_booted_stats(ax_element, y_samps, x, label, color, linewidth=2, alpha=0.3):
    """
    Wrapper for plotting the mean and 95% confidence intervals for bootstrap samples of a function on the y-axis
    """
    av = np.mean(y_samps, axis=0)
    lower = np.percentile(y_samps, 2.5, axis=0)
    upper = np.percentile(y_samps, 97.5, axis=0)
    ax_element.plot(x, av, color=color, label=label, linewidth=linewidth)
    ax_element.fill_between(x, lower, upper, alpha=alpha, color=color)


#------ Volumetric analysis tools --------#
def _fill_sphere(coord, grid, edges, spacing, radius) :
  """
  Fill a grid using spherical smoothing

  Parameters
  ----------
  coord : Numpy array
    the Cartesian coordinates to put on the grid
  grid  : Numpy array
    the 3D grid. Will be modified
  edges : list of Numpy array
    the edges of the grid
  spacing : float
    the grid spacing
  radius  : float
    the radius of the smoothing
  """
  # Maximum coordinate
  maxxyz = np.minimum(coord + radius, np.array([edges[0][-1], edges[1][-1], edges[2][-1]]))

  # Iterate over the sphere
  rad2 = radius**2
  x = max(coord[0] - radius,edges[0][0])
  while x <= maxxyz[0]:
    y = max(coord[1] - radius,edges[1][0])
    while y <= maxxyz[1]:
      z = max(coord[2] - radius, edges[2][0])
      while z <= maxxyz[2]:
        # Check if we are on the sphere
        r2 = (x - coord[0])**2 + (y - coord[1])**2 + (z - coord[2])**2  
        if r2 <= rad2:
          # Increase grid with one
          v = _voxel(np.array([x, y, z]), edges)
          grid[v[0], v[1], v[2]] += 1
        z += spacing
      y += spacing
    x += spacing

def _init_grid(xyz, spacing, padding) :
  """
  Initialize a grid based on a list of x,y,z coordinates

  Parameters
  ----------
  xyz  : Numpy array
    Cartesian coordinates that should be covered by the grid
  spacing : float
    the grid spacing
  padding : float
    the space to add to minimum extent of the coordinates

  Returns
  -------
  Numpy array
    the grid
  list of Numpy arrays
    the edges of the grid
  """

  origin = np.floor(xyz.min(axis=0)) - padding
  tr = np.ceil(xyz.max(axis=0)) + padding
  length = tr - origin
  shape = np.array([int(l/spacing + 0.5) + 1 for l in length], dtype=int)
  grid = np.zeros(shape)
  edges = [np.linspace(origin[i], tr[i], shape[i]) for i in range(3)]
  return grid, edges

def _voxel(coord, edges) :
  """
  Wrapper for the numpy digitize function to return the grid coordinates
  """
  return np.array([np.digitize(coord, edges[i])[i] for i in range(3)],dtype=int) - 1


def calc_solvent_accessible_volume(xyz, radii, solvent_rad=1.4, spacing=0.5):
    """
    Calculate the solvent accessible volume of a set of residues using spherical smoothing

    Parameters
    ----------
    traj : Mdtraj trejectory object
      the trajectory containing the residues of interest
    resnames  : list or tuple of strings
      the names of the residues whose volume will be calculated
    frame : integer
      the frame from which the positions will be taken from
    solvent_rad : float
      the radius of the solvent in nm. Arithmetic mean with residue will be used for volume calculation.
      If zero, only the vdW radii of the atoms will be used.
    spacing : float
      the grid spacing in nanometers

    Returns:
    -------
    volume : float
      the solvent accessible volume in nm^3 of the listed residues
    """
    grid, edges = _init_grid(xyz, spacing, 0)

    # The effective radii by taking into account the solvent.
    eff_radii = radii + solvent_rad

    for i in range(len(radii)):
        _fill_sphere(xyz[i], grid, edges, spacing, eff_radii[i])

    filled_grid = (grid >=1) * 1.0
    volume = np.sum(filled_grid) * (spacing ** 3.)

    return volume, filled_grid

def get_hueristic_salt_numbers(pdb_file, target_conc, ncounterions=0, mol_resnames=[]):
    """
    Calculate the number of salt pairs to insert into a simulation using 3 heuristic schemes that use:
        1. The volume of the periodic cell
        2. The solvent volume, given by the difference between the cell volume and macromolecule volume.
        3. The ratio of the number of water to salt pairs, assuming a water concentration of 55.5M.

    The macromocule volume is the sum of DNA nucleotides, proteins residues, and any residues that have been supplied
    by the user. The residue names of the proteins and DNA nucleotides do not need to be supplied.

    Parameters
    ----------
    pdb_file: str
        the name of the pdb file that contains the initial structure of the simulation.
    target_conc: float
        the required concentration of salt in M (not mM).
    ncounterions: int
        the number of neutralizing counterions that have been added. These will be deducted from the number of water
        molecules.
    mol_resnames: list
        A list of strings containing the residue names of small molecules.

    Returns
    -------
    nsalt_cell_vol_guess: int
        Number of salt pairs estimated using the total volume of periodic cell.
    nsalt_sol_vol_guess: int
        Number of salt pairs estimated using the volume of the solvent.
    nsalt_ratio_guess: int
        Number of salt pairs estimated via the ratio of salt pairs to water molecules.
    """

    traj = mdtraj.load(pdb_file)

    ### 1: Estimate using the volume of the periodic cell:
    cell_vol = traj.unitcell_lengths.cumprod()[-1]
    nsalt_cell_vol_guess = int(np.floor(target_conc * cell_vol / 1.66054))

    ### 2: Estimate the initial of the solvent:
    # Get the coordinates of DNA and protein molecules.
    dna_resnames = ['DA', 'DT', 'DC', 'DG', 'DG3', 'DC5']
    prot_resnames = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HID', 'HIP', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    macro_resnames = dna_resnames + prot_resnames + mol_resnames
    mol_indices = [atom.index for atom in traj.topology.atoms if ((atom.residue.name in macro_resnames))]

    if len(mol_indices) > 0:
        xyz_mol = traj.xyz[0][mol_indices, :]

        # Get the approximate radii of the macromolecule volume. (Mdtraj uses the Bondi radii)
        radii = np.array([atom.element.radius for atom in traj.topology.atoms if ((atom.residue.name in macro_resnames))])

        # Caculate the solvent accessible volume of the macromolecule.
        spacing = 0.05 # nm.
        solvant_radius = 0.14  # nm. Half of the oxygen-oxygen distance in the first solvation shell.
        mol_vol, grid = calc_solvent_accessible_volume(xyz_mol, radii, solvent_rad=solvant_radius, spacing=spacing)
        print("Volume of macromolecule = {0:.1f}".format(mol_vol))
    else:
        print('No macromolecules found in PDB. The solvent volume is the same as the total volume.')
        mol_vol = 0.0

    sol_vol = cell_vol - mol_vol
    #sol_vol = tools.calc_solvent_accessible_volume(traj, water_resname, frame=0, solvent_rad=0.0, spacing=0.04)
    nsalt_sol_vol_guess = int(np.floor(target_conc * sol_vol / 1.66054))

    ### 3: Estimate using the ratio of water molecules and salt pairs:
    nwaters = len(traj.topology.select_atom_indices('water')) - ncounterions
    water_conc = 55.4  # The value used in OpenMM.
    nsalt_ratio_guess = int(np.floor(nwaters * target_conc / water_conc))

    return nsalt_cell_vol_guess, nsalt_sol_vol_guess, nsalt_ratio_guess

def get_heurstic_concentrations(pdbfile, ncfile, target_conc):
    """
    Return approximations to the number of salt molecules that would be added to a system in fixed salt-fraction
    simulations.

    Parameters
    ----------
    pdbfile: str
        the name of the initial PDB file of the simulation. Used to estimate volumes.
    ncfile: str
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
    ncfile = Dataset(ncfile, 'r')
    volume = ncfile.groups['Sample state data']['volume'][:]
    nspecies = ncfile.groups['Sample state data']['species counts'][:]
    ncfile.close()

    ### Seeing how many water molecules are available to be turned into salt.
    # The number of neutralizing counterions
    ncation = nspecies[0, 1]
    nanion = nspecies[0, 2]
    ncounterions = int(abs(nanion - ncation))

    N_cell, N_sol, N_ratio = get_hueristic_salt_numbers(pdbfile, target_conc, ncounterions)

    # Converting all to concentrations to mM
    factor = 1660.54  # Converts number /nm^3 to mM
    return factor * N_cell/volume, factor * N_sol/volume, factor * N_ratio/volume

def writeDX(grid, origin, spacing, filename) :
  """
  Write the grid to file in DX-format

  Parameters
  ----------
  grid : Numpy array
    the 3D grid
  origin : NumpyArray
    the bottom-left coordinate of the grid
  spacing  : float
    the grid spacing
  filename : string
     the name of the DX file
  """
  f = open(filename, 'w')
  f.write("object 1 class gridpositions counts %5d%5d%5d\n"%(grid.shape[0],grid.shape[1],grid.shape[2]))
  f.write("origin %9.4f%9.4f%9.4f\n"%(origin[0],origin[1],origin[2]))
  f.write("delta %10.7f 0.0 0.0\n"%spacing)
  f.write("delta 0.0 %10.7f 0.0\n"%spacing)
  f.write("delta 0.0 0.0 %10.7f\n"%spacing)
  f.write("object 2 class gridconnections counts %5d%5d%5d\n"%(grid.shape[0],grid.shape[1],grid.shape[2]))
  f.write("object 3 class array type double rank 0 items  %10d data follows\n"%(grid.shape[0]*grid.shape[1]*grid.shape[2]))
  cnt = 0
  for x in range(grid.shape[0]) :
    for y in range(grid.shape[1]) :
      for z in range(grid.shape[2]) :
        f.write("%19.10E"%grid[x,y,z])
        cnt = cnt + 1
        if cnt >= 3 :
          cnt = 0
          f.write("\n")
  if cnt > 0 : f.write("\n")
  f.write('attribute "dep" string "positions"\n')
  f.write('object "regular positions regular connections" class field\n')
  f.write('component "positions" value 1\n')
  f.write('component "connections" value 2\n')
  f.write('component "data" value 3\n')
  f.close()
