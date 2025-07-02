import os 
print('test1')
for fn in os.listdir():
    if fn[-4:]=='.out':
        os.remove(fn)
print('test2')
from Procs import SlurmStartup
Handler=SlurmStartup(restart=False)

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import ase
    
class OTF_Calcualator(Calculator):
    implemented_properties = ['energy', 'forces','stress']
    def __init__(self, req_handler):
        super().__init__()
        self.req_handler = req_handler

    def calculate(self, atoms=None, properties=['energy', 'forces','stress'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        atoms,energy,forces,stress,E_uncert,F_uncert,S_uncert = self.req_handler(atoms,IncludeStress=True)
        if stress.shape == (1,3,3):
            stress=stress[0]

        # Store the results in the `_results` dictionary
        self.results['energy'] = energy/23.0609
        self.results['forces'] = forces/23.0609
        self.results['stress'] = stress/23.0609

calc=OTF_Calcualator(Handler)

from ase.md.langevin import Langevin
from ase import units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase import io
from ase.md import MDLogger

from ase import Atoms
from ase.spacegroup import crystal
from ase.build import make_supercell
from ase.md.verlet import VelocityVerlet


# Create the supercell
supercell = io.read('POSCAR')

# Define simulation parameters
initial_temp =  1500  # Start temperature
time_step = 1.5 * units.fs  # Time step in femtoseconds
n_steps = 20000  # Number of MD steps

# Set initial velocities
MaxwellBoltzmannDistribution(supercell, temperature_K=initial_temp)
supercell.set_calculator(calc)  
import numpy as np
mask=np.array([[1,0,0],[0,1,0],[0,0,1]])
supercell.set_cell(supercell.cell*mask)
from ase.optimize import BFGS

dyn = Langevin(supercell, timestep=0.5*time_step,temperature_K=1500, friction=5e-1)
dyn.run(1500)

dyn=VelocityVerlet(supercell,timestep=time_step)

# Save trajectory
traj = Trajectory("proton.traj", "w", supercell)
dyn.attach(traj.write, interval=1)  # Save every 100 steps

# Logging
dyn.attach(MDLogger(dyn, supercell, "npt.log", header=True, stress=True, peratom=True), interval=100)



# Run simulation
dyn.run(n_steps)

