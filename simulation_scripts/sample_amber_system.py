from simtk import openmm, unit
from openmmtools import testsystems
from simtk.openmm import app
from saltswap import wrappers
from openmmtools import integrators
import saltswap.record as Record
from time import time

"""
Sample the configurations and salt concentrations of system using an AMBER topology. The test system must contain
explicit water molecules.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an osmostated simulation using AMBER topology files.")
    parser.add_argument('--prmtop', type=str, help='The filename of the AMBER input topology')
    parser.add_argument('--inpcrd', type=str, help='The filename of the AMBER input coordinates', default=None)
    parser.add_argument('--ipdb', type=str,
                        help='The filename of the input PDB structure. Will overwrite inpcrd', default=None)
    parser.add_argument('-o','--out', type=str,
                        help="the naming scheme of the output results, default=out", default="out")
    parser.add_argument('-c','--conc', type=float,
                        help="the macroscopic salt concentration in M, default=0.2", default=0.2)
    parser.add_argument('-i','--iterations', type=int,
                        help="the number of iterations of MD and saltswap moves, default=2500", default=2500)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per iteration, default=2000", default=2000)
    parser.add_argument('--save_freq', type=int,
                        help="the frequency with which to save the data", default=4)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators in femtoseconds, default=2.0', default=2.0)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=1000", default=1000)
    parser.add_argument('--npert', type=int,
                        help="the number of ncmc perturbation kernels, default=1000", default=1000)
    parser.add_argument('--nprop', type=int,
                        help="the number of propagation steps per perturbation kernels, default=10", default=10)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    parser.add_argument('--save_configs', action='store_true',
                        help="whether to save the configurations of the box of water, default=True", default=True)
    parser.add_argument('--water_name', type=str,
                        help="the residue name of the water molecules, default=HOH", default='HOH')
    parser.add_argument('--fix_salt', action='store_true',
                        help='Whether to NOT sample over salt numbers.', default=False)

    args = parser.parse_args()

    #-------- Create the system using default SaltSwap non-bonded parameters ---------#
    nonbondedCutoff = 10 * unit.angstrom
    use_dispersion_correction = True
    switch_width = 1.5 * unit.angstrom
    ewaldErrorTolerance = 1E-4

    # Load the topology
    prmtop = app.AmberPrmtopFile(args.prmtop)
    topology = prmtop.topology

    # Initialize system
    system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=app.PME, rigidWater=True,
                                         nonbondedCutoff=nonbondedCutoff, hydrogenMass=None)

    # Set dispersion correction use.
    forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
              range(system.getNumForces())}
    forces['NonbondedForce'].setUseDispersionCorrection(True)
    forces['NonbondedForce'].setEwaldErrorTolerance(1E-4)

    # Set switch width
    forces['NonbondedForce'].setUseSwitchingFunction(True)
    forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

    # Load coordinates
    if args.ipdb is not None:
        pdbfile = app.PDBFile(args.ipdb)
        positions = pdbfile.getPositions()
    else:
        inpcrd = app.AmberInpcrdFile(args.inpcrd)
        positions = inpcrd.getPositions(asNumpy=True)
        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

    #--------- Osmostat simulation parameters and setup ---------#
    # Setting the parameters of the simulation
    timestep = args.timestep * unit.femtoseconds
    npert = args.npert
    nprop = args.nprop

    # Fixed simulation parameters
    splitting = 'V R O R V'
    temperature = 300.0 * unit.kelvin
    collision_rate = 1.0 / unit.picoseconds
    pressure = 1.0 * unit.atmospheres
    salt_concentration = args.conc * unit.molar

    # Add the barostat.
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    # Create the compound integrator
    langevin = integrators.LangevinIntegrator(splitting=splitting, temperature=temperature, timestep=timestep,
                                              collision_rate=collision_rate, measure_shadow_work=False,
                                              measure_heat=False)
    ncmc_langevin = integrators.ExternalPerturbationLangevinIntegrator(splitting=splitting, temperature=temperature,
                                                                       timestep=timestep, collision_rate=collision_rate,
                                                                       measure_shadow_work=False, measure_heat=False)
    integrator = openmm.CompoundIntegrator()
    integrator.addIntegrator(langevin)
    integrator.addIntegrator(ncmc_langevin)

    # Create context
    if args.platform == 'CUDA':
        platform = openmm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        properties = {'CudaPrecision': 'mixed'}
        context = openmm.Context(system, integrator, platform, properties)
    elif args.platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(system, integrator, platform, properties)
    elif args.platform == 'CPU':
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(system, integrator, platform)
    else:
        raise Exception('Platform name {0} not recognized.'.format(args.platform))

    context.setPositions(positions)
    context.setVelocitiesToTemperature(temperature)

    # Create the swapper object for the insertion and deletion of salt
    salinator = wrappers.Salinator(context=context, system=system, topology=topology,
                                   ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
                                   pressure=pressure, temperature=temperature, npert=npert, nprop=nprop,
                                   water_name=args.water_name)

    # Neutralize the system and initialize the number of salt pairs.
    salinator.neutralize()
    salinator.initialize_concentration()

    # Minimize the system
    openmm.LocalEnergyMinimizer.minimize(context)

    # Thermalize the system with the initial number of salt pairs fixed.
    langevin.step(args.equilibration)

    # Create the NetCDF file for non-configuration simulation data
    filename = args.out + '.nc'
    creator = Record.CreateNetCDF(filename)
    simulation_control_parameters = {'timestep': timestep, 'splitting': splitting, 'collision_rate': collision_rate}
    ncfile = creator.create_netcdf(salinator.swapper, simulation_control_parameters)
    var = ncfile.groups['Sample state data'].createVariable('time', 'f4', ('iteration'), zlib=True)
    var.unit = 'seconds'

    if args.save_configs:
        # Create PDB file to view with the (binary) dcd file.
        positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
        pdbfile = open(args.out + '.pdb', 'w')
        app.PDBFile.writeHeader(topology, file=pdbfile)
        app.PDBFile.writeModel(topology, positions, file=pdbfile, modelIndex=0)
        pdbfile.close()

        # Create a DCD file system configurations
        dcdfile = open(args.out + '.dcd', 'wb')
        dcd = app.DCDFile(file=dcdfile, topology=topology, dt=timestep)

    # The actual simulation
    k = 0
    for iteration in range(args.iterations):
        # Propagate configurations and salt concentrations
        t0 = time()
        langevin.step(args.steps)
        if not args.fix_salt:
            salinator.update(nattempts=1)
        iter_time = time() - t0
        if iteration % args.save_freq == 0:
            # Record the simulation data
            Record.record_netcdf(ncfile, context, salinator.swapper, k, attempt=0, sync=False)
            ncfile.groups['Sample state data']['time'][k] = iter_time
            ncfile.sync()
            if args.save_configs:
                # Record the simulation configurations
                state = context.getState(getPositions=True, enforcePeriodicBox=True)
                positions = state.getPositions(asNumpy=True)
                dcd.writeModel(positions=positions)
            k += 1
