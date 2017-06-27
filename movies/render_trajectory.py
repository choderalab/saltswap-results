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
import os
import os.path
from pymol import cmd
from pymol import util
import mdtraj

# try to keep PyMOL quiet
#cmd.set('internal_gui', 0)
#cmd.feedback("disable","all","actions")
#cmd.feedback("disable","all","results")

#=============================================================================================
# PARAMETERS
#=============================================================================================

netcdf_filename = '../200mM/out2.nc'
trajectory_filename = '../200mM/out2.dcd'
reference_pdb_filename = '../200mM/out2.pdb'
png_dir = 'png'

if not os.path.exists(png_dir):
    print("Creating directory '%s'..." % png_dir)
    os.makedirs(png_dir)

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# MAIN
#=============================================================================================

#import __main__
#__main__.pymol_argv = [ 'pymol', '-qc']
#import pymol
#pymol.finish_launching()

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
# Find water molecules
waters = [ residue for residue in traj.topology.residues if residue.is_water ]
water_oxygen_indices = traj.topology.select_atom_indices('water')

# Reset
cmd.rewind()
cmd.delete('all')
cmd.reset()

# Load PDB file into PyMOL
cmd.load(reference_pdb_filename, 'system')
cmd.hide('all')
cmd.select('solute', '(not resn WAT) and (not hydrogen)')
cmd.select('water', 'resn WAT')
cmd.deselect()
cmd.show('cartoon', 'solute')
cmd.color('white', 'solute')

# Remove hydrogens
#cmd.remove('hydrogens')

# speed up builds
#cmd.set('defer_builds_mode', 3)
#cmd.set('cache_frames', 0)
cmd.set('async_builds', 1)

cmd.set('ray_transparency_contrast', 3.0)
cmd.set('ray_transparency_shadows', 0)

model = cmd.get_model('system')
#for atom in model.atom:
#    print "%8d %4s %3s %5d %8.3f %8.3f %8.3f" % (atom.index, atom.name, atom.resn, int(atom.resi), atom.coord[0], atom.coord[1], atom.coord[2])

#pymol.finish_launching()

cmd.viewport(640,480)
#niterations = 10 # DEBUG

# Load trajectory
cmd.load_traj(trajectory_filename, object='system')

# Align all states
cmd.intra_fit('solute')

# Zoom viewport
cmd.zoom('solute')
#cmd.zoom('resi 45','+20') # zoom
#cmd.show('sticks', '(sidechain and not hydrogen) within 6 of resi 44')
#cmd.set('cartoon_side_chain_helper', '1')
#util.cbaw('solute')
#cmd.orient('complex')

#cmd.hide('all')
#cmd.rewind()

#md.set('transparency', 0.65)
#cmd.set('surface_mode', 3)
#cmd.set('surface_color', 'white')

# Create one-to-one mapping between states and frames.
cmd.mset("1 -%d" % nframes)

#cmd.zoom('ligand')
#cmd.orient('ligand')
#cmd.turn('x', -90)

# Delete first frame
cmd.mdelete("1")

# Render movie
#frame_prefix = 'frames/frame'
#cmd.set('ray_trace_frames', 1)
#cmd.set('ray_trace_frames', 0) # DEBUG
#nframes = 1
for frame in range(nframes):
    print "rendering frame %04d / %04d" % (frame+1, nframes)
    cmd.frame(frame+1)
    # Show only ions
    cmd.hide('spheres', 'water')
    # Determine oxygen indices for cations and anions
    cation_indices = [ water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==1 ]
    anion_indices = [ water_oxygen_indices[water_index] for water_index in range(nwaters) if identities[frame, water_index]==2 ]
    cation_selection = 'index ' + '+'.join([str(index) for index in cation_indices])
    anion_selection = 'index ' + '+'.join([str(index) for index in anion_indices])
    cmd.show('spheres', cation_selection)
    cmd.show('spheres', anion_selection)
    cmd.color('yellow', cation_selection)
    cmd.color('green', anion_selection)
    filename = os.path.join(png_dir, 'frame%05d.png' % frame)
    print(filename)
    cmd.png(filename)
