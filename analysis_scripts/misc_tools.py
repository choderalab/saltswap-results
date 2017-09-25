import numpy as np
from netCDF4 import Dataset
from scipy.stats import norm

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
        # If one wants to include the ionic strength of the biomolecule:
        # ionic_strength.append((ac + cc + bc*(biomol_charge**2))/2.0)


    return 1000*np.array(salt_conc), 1000*np.array(ionic_strength), 1000*np.array(cation_conc), 1000*np.array(anion_conc), np.array(salt_number), ntotal_species


def _fill_gauss(coord,grid,edges,spacing,std) :
  """
  Fill a grid using Gaussian smoothing

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
  std  : float
    the sigma of the Gaussian distribution
  """
  # Maximum coordinate
  maxxyz = np.minimum(coord + 3*std,np.array([edges[0][-1],edges[1][-1],edges[2][-1]]))

  # Iterater over 3 standard deviations
  x = max(coord[0] - 3*std,edges[0][0])
  gx = norm.pdf(x,coord[0],std)
  while x <= maxxyz[0] :
    y = max(coord[1] - 3*std,edges[1][0])
    gy = norm.pdf(y,coord[1],std)
    while y <= maxxyz[1] :
      z = max(coord[2] - 3*std,edges[2][0])
      gz = norm.pdf(z,coord[2],std)
      while z <= maxxyz[2] :
        # Increase the grid with a Gaussian probability density
        v = _voxel(np.array([x,y,z]),edges)
        grid[v[0],v[1],v[2]] = grid[v[0],v[1],v[2]] + gx*gy*gz

        z = z + spacing
        gz = norm.pdf(z,coord[2],std)
      y = y + spacing
      gy = norm.pdf(y,coord[1],std)
    x = x + spacing
    gx = norm.pdf(x,coord[0],std)

def _fill_sphere(coord,grid,edges,spacing,radius) :
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

def _init_grid(xyz,spacing,padding) :
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

  origin  = np.floor(xyz.min(axis=0))-padding
  tr      = np.ceil(xyz.max(axis=0))+padding
  length  = tr-origin
  shape  =  np.array([int(l/spacing + 0.5) + 1 for l in length],dtype=int)
  grid    = np.zeros(shape)
  edges  = [np.linspace(origin[i],tr[i],shape[i]) for i in range(3)]
  return grid,edges

def _voxel(coord,edges) :
  """
  Wrapper for the numpy digitize function to return the grid coordinates
  """
  return np.array([np.digitize(coord,edges[i])[i] for i in range(3)],dtype=int) - 1

def writeDX(grid,origin,spacing,filename) :
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
