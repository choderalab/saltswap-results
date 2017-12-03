#!/usr/bin/env python

"""
Render a saltswap trajectory with PyMOL
"""

#=============================================================================================
# IMPORTS
#=============================================================================================

import numpy as np
#import Scientific.IO.NetCDF
import netCDF4 as NetCDF
import os, shutil
import os.path
from pymol import cmd, util
import mdtraj

# try to keep PyMOL quiet
#cmd.set('internal_gui', 0)
#cmd.feedback("disable","all","actions")
#cmd.feedback("disable","all","results")

#=============================================================================================
# PARAMETERS
#=============================================================================================

solute = 'protein' # 'dna' or 'protein'
prefix = '../testsystems/dhfr/200mM' # '100mM', '200mM', or 'dna_dodecamer'
png_dir = 'png' # BE CAREFUL: This directory will be removed every time you run this

width = 640
height = 480

netcdf_filename = '%s/out1.nc' % prefix
trajectory_filename = '%s/out1.dcd' % prefix
reference_pdb_filename = '%s/out1.pdb' % prefix

if os.path.exists(png_dir):
    shutil.rmtree(png_dir)
os.makedirs(png_dir)

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# MAIN
#=============================================================================================

# Image trajectory
print('Reading trajectory...')
traj = mdtraj.load(trajectory_filename, top=reference_pdb_filename)
print('Superimposing biomolecule...')
# Align all states
if solute == 'dna':
    align_atoms = traj.topology.select('name == P') # DNA
else:
    align_atoms = traj.topology.select('protein')
print('Will align on %d atoms...' % len(align_atoms))
traj.superpose(traj, frame=0, atom_indices=align_atoms)
print('Imaging trajectory...')
traj.image_molecules()
print('Writing trajectory...')
trajectory_temporary_filename = 'out.dcd'
traj.save(trajectory_temporary_filename)

# Required for python 2.x
import __main__
__main__.pymol_argv = [ 'pymol', '-qc']
import pymol
pymol.finish_launching()

# Read chemical identities
print("Reading identities from '%s'..." % netcdf_filename)
ncfile = NetCDF.Dataset(netcdf_filename, 'r')
identities = ncfile.groups['Sample state data']['identities'][:,:]
print(identities.shape)
[nframes, nwaters] = identities.shape
print('There are %d frames for %d waters' % (nframes, nwaters))
ncfile.close()

# Read PDB file into MDTraj
print('Reading trajectory into mdtraj...')
traj = mdtraj.load(reference_pdb_filename)
natoms = sum([1 for atom in traj.topology.atoms])
# Find water molecules
water_oxygen_indices = traj.topology.select_atom_indices('water') + 1 # pymol indices start from 1

# Reset
cmd.rewind()
cmd.delete('all')
cmd.reset()

# Load PDB file into PyMOL
cmd.set('suspend_updates', 'on')
cmd.set('retain_order', 1)
cmd.load(reference_pdb_filename, object='system')
cmd.hide('all')
print('selecting solute...')
print(cmd.select('solute', '(not resn WAT) and (not resn HOH) and (not hydrogen)'))
cmd.select('water', 'resn WAT')
cmd.deselect()

#cmd.show('cartoon', 'solute')
#cmd.color('white', 'solute')

cmd.orient('solute')
util.cbaw('solute')
if solute == 'dna':
    cmd.show('sticks', 'solute') # DNA
else:
    cmd.show('cartoon', 'solute') # DNA

# speed up builds
cmd.set('defer_builds_mode', 3)
cmd.set('cache_frames', 0)
cmd.cache('disable')
cmd.set('async_builds', 1)

cmd.set('ray_transparency_contrast', 3.0)
cmd.set('ray_transparency_shadows', 0)

model = cmd.get_model('system')
#for atom in model.atom:
#    print "%8d %4s %3s %5d %8.3f %8.3f %8.3f" % (atom.index, atom.name, atom.resn, int(atom.resi), atom.coord[0], atom.coord[1], atom.coord[2])

#pymol.finish_launching()

cmd.viewport(width,height)
#niterations = 10 # DEBUG

# Load trajectory
cmd.load_traj(trajectory_temporary_filename, object='system')

# Align all states
if solute == 'dna':
    cmd.intra_fit('name P') # DNA
else:
    cmd.intra_fit('solute') # protein

# Zoom viewport
cmd.zoom('solute')
cmd.orient('solute')

# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % nframes)


# Delete first frame
cmd.mdelete("1")

# Render movie
cmd.set('ray_trace_frames', 0)
#nframes = 1
for frame in range(nframes):
    print("rendering frame %04d / %04d" % (frame+1, nframes))
    cmd.set('suspend_updates', 'on')
    cmd.frame(frame+1)
    # Show only ions
    cmd.hide('spheres', 'all')
    # Determine oxygen indices for cations and anions
    cation_indices = [ water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==1 ]
    anion_indices = [ water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==2 ]
    #atom_identities = np.zeros([natoms], np.int32)
    #atom_identities[cation_indices] = 1
    #atom_identities[anion_indices] = 2
    #myspace = {'atom_identities': [0] + [int(x) for x in atom_identities]}
    #cmd.alter('(all)', 'p.identity = atom_identities[ID]', space=myspace)
    cation_selection = 'id ' + '+'.join([str(index) for index in cation_indices])
    anion_selection = 'id ' + '+'.join([str(index) for index in anion_indices])
    cmd.show('spheres', cation_selection)
    cmd.show('spheres', anion_selection)
    cmd.color('yellow', cation_selection)
    cmd.color('green', anion_selection)
    #cmd.show('spheres', 'p.identity > 0')
    #cmd.color('yellow', 'p.identity = 1')
    #cmd.color('green', 'p.identity = 2')

    #if solute == 'dna':
    #    cmd.hide('spheres', 'water beyond 6 of solute')
    filename = os.path.join(png_dir, 'frame%05d.png' % frame)
    print(filename)
    cmd.set('suspend_updates', 'off')
    cmd.png(filename, width, height, ray=0)
