from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app

from saltswap.swapper import Swapper
from openmmtools import integrators
import saltswap.record as Record

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a saltswap simulation on a box of water.")
    parser.add_argument('-o','--out', type=str,
                        help="the naming scheme of the output results." ,default="out")
    parser.add_argument('-b','--box_edge', type=float,
                        help="length of the water box edge in Angstroms" ,default=20.0)
    parser.add_argument('-u','--deltachem', type=float,
                        help="the applied chemical potential in thermal units, default=335", default=335.0)
    parser.add_argument('-i','--iterations', type=int,
                        help="the number of cycles between MD and MCMC salt-water swaps, default=200", default=200)
    parser.add_argument('-s','--steps', type=int,
                        help="the number of MD steps per cycle, default=2000", default=2000)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators, default=2.0', default=2.0)
    parser.add_argument('-e','--equilibration', type=int,
                        help="the number of equilibration steps, default=1000", default=1000)
    parser.add_argument('--npert', type=int,
                        help="the number of _ncmc perturbation kernels, default=1000", default=1000)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    # Setting the parameters of the simulation
    timestep = args.timestep * unit.femtoseconds
    box_edge = args.box_edge * unit.angstrom
    npert = args.npert * unit.angstrom

    # Fixed simulation parameters
    splitting = 'V R R R R O R R R R V'
    temperature = 300. * unit.kelvin
    collision_rate = 1. / unit.picoseconds
    pressure = 1. * unit.atmospheres

    # Make the water box test system with a fixed pressure
    wbox = WaterBox(box_edge=box_edge, nonbondedMethod=app.PME, cutoff=9 * unit.angstrom, ewaldErrorTolerance=1E-4)
    wbox.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    # Create the compound integrator
    langevin = integrators.LangevinIntegrator(splitting=splitting, temperature=temperature, timestep=timestep,
                                              collision_rate=collision_rate)
    ncmc_langevin = integrators.ExternalPerturbationLangevinIntegrator(splitting=splitting, temperature=temperature,
                                                                       timestep=timestep, collision_rate=collision_rate)
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

    # Create the swapper object for the insertion and deletion of salt
    salinator = Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=args.deltachem,
                        ncmc_integrator=ncmc_langevin, pressure=pressure, npert=npert, nprop=1)

    # Thermalize the system
    langevin.step(args.equilibration)

    # Create the netcdf file for non-configuration simulation data
    filename = args.out + '.nc'
    creator = Record.CreateNetCDF(filename)
    simulation_control_parameters = {'timestep': timestep, 'splitting': splitting, 'box_edge': box_edge,
                                     'collision_rate': collision_rate}
    ncfile = creator.create_netcdf(salinator, simulation_control_parameters)

    # Create PDB file to view with the (binary) dcd file.
    positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
    pdbfile = open(args.out + '.pdb', 'w')
    app.PDBFile.writeHeader(wbox.topology, file=pdbfile)
    app.PDBFile.writeModel(wbox.topology, positions, file=pdbfile, modelIndex=0)
    pdbfile.close()

    # Create a DCD file system configurations
    dcdfile = open(args.out + '.dcd', 'wb')
    dcd = app.DCDFile(file=dcdfile, topology=wbox.topology, dt=args.timestep)

    # The actual simulation part
    for iteration in range(args.iterations):
        # Propagate configurations and salt concentrations
        langevin.step(args.steps)
        salinator.update(context, nattempts=1)
        # Record the simulation data
        Record.record_netcdf(ncfile, context, salinator, iteration, attempt=0, sync=True)
        # Record the simulation configurations
        positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
        dcd.writeModel(positions=positions)