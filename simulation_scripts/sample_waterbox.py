from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app

from saltswap import wrappers
from openmmtools import integrators
import saltswap.record as Record
from time import time

"""
Sample the configurations and salt concentrations on a box of water by specifying the macroscopic salt concentration.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a saltswap simulation on a box of water.")
    parser.add_argument('-o','--out', type=str,
                        help="the naming scheme of the output results, default=out", default="out")
    parser.add_argument('-b','--box_edge', type=float,
                        help="length of the water box edge in Angstroms, default=30", default=30.0)
    parser.add_argument('-c','--conc', type=float,
                        help="the macroscopic salt concentration in M, default=0.2", default=0.2)
    parser.add_argument('-i','--iterations', type=int,
                        help="the number of iterations of MD and saltswap moves, default=7500", default=8000)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per iteration, default=2000", default=2000)
    parser.add_argument('--save_freq', type=int,
                        help="the frequency with which to save the data, default=1", default=1)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators in femtoseconds, default=2.0', default=2.0)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=5000", default=5000)
    parser.add_argument('--model', type=str, choices=['tip3p','tip4pew'],
                        help="the water model, default=tip4ew", default='tip4pew')
    parser.add_argument('--npert', type=int,
                        help="the number of ncmc perturbation kernels, default=1000", default=1000)
    parser.add_argument('--nprop', type=int,
                        help="the number of propagation steps per perturbation kernels, default=10", default=10)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    parser.add_argument('--save_configs', action='store_true',
                        help="whether to save the configurations of the box of water, default=False", default=False)
    args = parser.parse_args()

    # Setting the parameters of the simulation
    timestep = args.timestep * unit.femtoseconds
    box_edge = args.box_edge * unit.angstrom
    npert = args.npert
    nprop = args.nprop

    # Fixed simulation parameters
    splitting = 'V R O R V'
    temperature = 300.*unit.kelvin
    collision_rate = 1./unit.picoseconds
    pressure = 1.*unit.atmospheres
    salt_concentration = args.conc * unit.molar

    # Make the water box test system with a fixed pressure
    wbox = WaterBox(box_edge=box_edge, model=args.model, nonbondedMethod=app.PME, cutoff=10*unit.angstrom, ewaldErrorTolerance=1E-4)
    wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

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
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif args.platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif args.platform == 'CPU':
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
    else:
        raise Exception('Platform name {0} not recognized.'.format(args.platform))

    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(temperature)

    # Create the salinator object for the insertion and deletion of salt
    salinator = wrappers.Salinator(context=context, system=wbox.system, topology=wbox.topology,
                                   ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
                                   pressure=pressure, temperature=temperature, npert=npert, nprop=nprop,
                                   water_name='HOH')

    # Neutralize the system and initialize the number of salt pairs.
    salinator.neutralize()
    salinator.initialize_concentration()

    # Thermalize the system
    langevin.step(args.equilibration)

    # Create the netcdf file for non-configuration simulation data
    filename = args.out + '.nc'
    creator = Record.CreateNetCDF(filename)
    simulation_control_parameters = {'timestep': timestep, 'splitting': splitting, 'box_edge': box_edge,
                                     'collision_rate': collision_rate}
    ncfile = creator.create_netcdf(salinator.swapper, simulation_control_parameters)
    var = ncfile.groups['Sample state data'].createVariable('time', 'f4', ('iteration'), zlib=True)
    var.unit = 'seconds'

    if args.save_configs:
        # Create PDB file to view with the (binary) dcd file.
        positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
        pdbfile = open(args.out + '.pdb', 'w')
        app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
        app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=0)
        pdbfile.close()

        # Create a DCD file system configurations
        dcdfile = open(args.out + '.dcd', 'wb')
        dcd = app.DCDFile(file=dcdfile, topology=wbox.topology, dt=timestep)

    k = 0
    # The actual simulation
    for iteration in range(args.iterations):
        # Propagate configurations and salt concentrations
        langevin.step(args.steps)
        t0 = time()
        salinator.update(nattempts=1)
        move_time = time() - t0
        if iteration % args.save_freq == 0:
            # Record the simulation data
            Record.record_netcdf(ncfile, context, salinator.swapper, k, attempt=0, sync=False)
            ncfile.groups['Sample state data']['time'][k] = move_time
            ncfile.sync()
            if args.save_configs:
                # Record the simulation configurations
                positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
                dcd.writeModel(positions=positions)
            k += 1