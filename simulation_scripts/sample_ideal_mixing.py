from simtk import openmm, unit
from openmmtools.testsystems import WaterBox
from simtk.openmm import app
import numpy as np
from saltswap.swapper import Swapper
from openmmtools import integrators
from time import time
"""
A script to sample ideal mixing and automatically analyze the results.

The nonbonded parameters of Na+ and Cl- will be given the same parameters as water, and a small box of water will be
simulated with the osmostat. The protocol work for an ideal swap should be as close to zero as possible for all transformations.
"""
def create_ideal_system(npert=50, nprop=1, deltachem=0.0, platform='CPU'):
    """
    Create small box of water that can impliment ideal mixing with the SaltSwap osmostat.

    Parameters
    ----------
    npert: int
        the number of NCMC perturbations
    nprop: int
        the number of Langevin propagation steps per NCMC perturbation.
    deltachem: float
        the difference in chemical potential between two water molecules and NaCl that has the same nonbonded parameters
        as two water molecules.
    platform: str
        The computational platform. Either 'CPU', 'CUDA', or 'OpenCL'.

    Returns
    -------
    ncmc_swapper: saltswap.swapper
        the driver that can perform NCMC exchanges
    langevin: openmmtools.integrator
        the integrator for equilibrium sampling.
    """
    # Setting the parameters of the simulation
    timestep = 2.0 * unit.femtoseconds
    box_edge = 25.0 * unit.angstrom
    splitting = 'V R O R V'
    temperature = 300.*unit.kelvin
    collision_rate = 1./unit.picoseconds
    pressure = 1.*unit.atmospheres

    # Make the water box test system with a fixed pressure
    wbox = WaterBox(box_edge=box_edge, model='tip3p', nonbondedMethod=app.PME, cutoff=10*unit.angstrom, ewaldErrorTolerance=1E-4)
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
    if platform == 'CUDA':
        platform = openmm.Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        properties = {'CudaPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        context = openmm.Context(wbox.system, integrator, platform, properties)
    elif platform == 'CPU':
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(wbox.system, integrator, platform)
    else:
        raise Exception('Platform name {0} not recognized.'.format(args.platform))

    context.setPositions(wbox.positions)
    context.setVelocitiesToTemperature(temperature)

    # Create the swapper object for the insertion and deletion of salt
    ncmc_swapper = Swapper(system=wbox.system, topology=wbox.topology, temperature=temperature, delta_chem=deltachem,
                        ncmc_integrator=ncmc_langevin, pressure=pressure, npert=npert, nprop=nprop)

    # Set the nonbonded parameters of the ions to be the same as water. This is critical for ideal mixing.
    ncmc_swapper.cation_parameters = ncmc_swapper.water_parameters
    ncmc_swapper.anion_parameters = ncmc_swapper.water_parameters
    ncmc_swapper._set_parampath()

    return context, ncmc_swapper, langevin

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run an ideal mixing simulation on a box of water and evaluate the "
                                                 "protocol work. This should be close to zero as possible.")
    parser.add_argument('-u','--deltachem', type=float,
                        help="the applied chemical potential in thermal units, default=0.0", default=0.0)
    parser.add_argument('-i','--iterations', type=int,
                        help="the number of iterations of MD and saltswap moves, default=10", default=10)
    parser.add_argument('--timestep', type=float,
                        help='the timestep of the integrators in femtoseconds, default=2.0', default=2.0)
    parser.add_argument('--npert', type=int,
                        help="the number of ncmc perturbation kernels, default=1000", default=1000)
    parser.add_argument('--nprop', type=int,
                        help="the number of propagation steps per perturbation kernels, default=10", default=10)
    parser.add_argument('--platform', type=str, choices=['CPU','CUDA','OpenCL'],
                        help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    # Create the system:
    context, ncmc_swapper, langevin = create_ideal_system(npert=args.nprop, nprop=args.nprop, deltachem=float(args.deltachem),
                                                          platform=args.platform)

    t0 = time()
    # Minimize the system
    openmm.LocalEnergyMinimizer.minimize(context)
    # Thermalize the system
    langevin.step(1000)
    print('Minimization and thermalization took {0:.1f} seconds'.format(time() - t0))

    protocol_work = np.zeros(args.iterations)
    t0 = time()
    for iteration in range(args.iterations):
        # Peform a ideal exchange
        ncmc_swapper.update(context, nattempts=1)
        protocol_work[iteration] = ncmc_swapper.cumulative_work[-1]
    print('Data collection for ideal mixing took {0:.1f} seconds'.format(time() - t0))

    print('Mean protocol work = {0} kT, with 95% of samples between {1} kT and {2} kT'.format(protocol_work.mean(), np.percentile(protocol_work, 2.5),  np.percentile(protocol_work, 97.5)))
    print('Maximum absolute value of protocol work = {0} kT'.format(np.max(np.abs(protocol_work))))
