from simtk import openmm, unit
from openmmtools import testsystems
from simtk.openmm import app
from saltswap import wrappers
from openmmtools import integrators
import numpy as np
from time import time
import gc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a saltswap simulation on a box of water.")
    parser.add_argument('-t','--testsystem', type=str,
                        help="the name of the openmmtools test system, default=DHFRExplicit", default='DHFRExplicit')
    parser.add_argument('-c','--conc', type=float,
                        help="the macroscopic salt concentration in M, default=0.2", default=0.2)
    parser.add_argument('-r','--repeats', type=int,
                        help="the number of benchmark repeats, default=5", default=5)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators in femtoseconds, default=2.0', default=2.0)
    parser.add_argument('--npert', type=int,
                        help="the number of ncmc perturbation kernels, default=10000", default=10000)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    parser.add_argument('--water_name', type=str,
                        help="the residue name of the water molecules, default=WAT", default='WAT')

    args = parser.parse_args()

    # Setting the parameters of the simulation
    timestep = args.timestep * unit.femtoseconds
    npert = args.npert

    def create_system(args, splitting):
        """
        Create a test system that's able to run saltswap.

        Parameters
        ----------
        args:
        splitting:
        """
        # Fixed simulation parameters
        temperature = 300.0 * unit.kelvin
        collision_rate = 1.0 / unit.picoseconds
        pressure = 1.0 * unit.atmospheres
        salt_concentration = args.conc * unit.molar

        # Get the test system and add the barostat.
        testobj = getattr(testsystems, args.testsystem)
        testsys = testobj(nonbondedMethod=app.PME, cutoff=10 * unit.angstrom, ewaldErrorTolerance=1E-4,
                          switch_width=1.5 * unit.angstrom)
        testsys.system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

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
            context = openmm.Context(testsys.system, integrator, platform, properties)
        elif args.platform == 'OpenCL':
            platform = openmm.Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            context = openmm.Context(testsys.system, integrator, platform, properties)
        elif args.platform == 'CPU':
            platform = openmm.Platform.getPlatformByName('CPU')
            context = openmm.Context(testsys.system, integrator, platform)
        else:
            raise Exception('Platform name {0} not recognized.'.format(args.platform))

        context.setPositions(testsys.positions)
        context.setVelocitiesToTemperature(temperature)

        # Create the swapper object for the insertion and deletion of salt
        salinator = wrappers.Salinator(context=context, system=testsys.system, topology=testsys.topology,
                                       ncmc_integrator=ncmc_langevin, salt_concentration=salt_concentration,
                                       pressure=pressure, temperature=temperature, npert=npert, water_name=args.water_name)

        # Neutralize the system and initialize the number of salt pairs.
        salinator.neutralize()
        salinator.initialize_concentration()

        return salinator, langevin, integrator

    print('Benchmark using', args.testsystem, 'with {0} NCMC perturbations with a 2 fs timestep'.format(args.npert))

    cor = [True, False]
    splitting = 'V R O R V'
    for c in cor:
        salinator, langevin, integrator = create_system(args, splitting)

        nbforce = salinator._get_nonbonded_force()
        nbforce.setUseDispersionCorrection(c)

        # Minimize the system
        openmm.LocalEnergyMinimizer.minimize(salinator.context)

        # Time insertion
        delta_ts = np.zeros(args.repeats)
        for i in range(args.repeats):
            t0 = time()
            salinator.update(nattempts=1)
            delta_ts[i] = time() - t0

        if args.repeats > 1:
            print('Splitting =', splitting, ', disp. correction =', c, 'time in seconds = {0:.2f} +/- {1:.2f}'.format(np.mean(delta_ts), np.std(delta_ts)/np.sqrt(args.repeats)))
        else:
            print('Splitting =', splitting, ', disp. correction =', c, ', time in seconds = ', delta_ts[0])

        del salinator.context, integrator, langevin, salinator
        gc.collect()

    print()
    cor = True
    splittings = ['V R R O R R V', 'V R R R O R R R V', 'V R R R R O R R R R V']
    for s in splittings:
        salinator, langevin, integrator = create_system(args, s)

        nbforce = salinator._get_nonbonded_force()
        nbforce.setUseDispersionCorrection(cor)

        # Minimize the system
        openmm.LocalEnergyMinimizer.minimize(salinator.context)

        # Time insertion
        delta_ts = np.zeros(args.repeats)
        for i in range(args.repeats):
            t0 = time()
            salinator.update(nattempts=1)
            delta_ts[i] = time() - t0
        if args.repeats > 1:
            print('Splitting =', s, ', disp. correction =', cor, ', time in seconds = {0:.2f} +/- {1:.2f}'.format(np.mean(delta_ts), np.std(delta_ts)/np.sqrt(args.repeats)))
        else:
            print('Splitting =', s, ', disp. correction =', cor, ', time in seconds = ', delta_ts[0])

        del salinator.context, integrator, langevin, salinator
        gc.collect()
