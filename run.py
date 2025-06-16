import os 
print('test1')
for fn in os.listdir():
    if fn[-4:]=='.out':
        os.remove(fn)
print('test2')

import yaml


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
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory
from ase import io
from ase.md import MDLogger

from ase import Atoms
from ase.spacegroup import crystal
from ase.build import make_supercell


# Create the supercell
supercell = io.read('POSCAR')

# Define simulation parameters
initial_temp = 300  # Start temperature
final_temp = 800  # Final temperature in Kelvin
pressure = 1.0 * units.bar  # Pressure in bar
time_step = 0.5 * units.fs  # Time step in femtoseconds
n_steps = 150000  # Number of MD steps
temp_ramp_steps = 50000  # Steps until temperatue jump
bulk_mod=160*units.GPa              # approximate bulk modulus in ev /A^3

# Set initial velocities at 300 K
MaxwellBoltzmannDistribution(supercell, temperature_K=initial_temp)
supercell.set_calculator(calc)  
import numpy as np
mask=np.array([[1,0,0],[0,1,0],[0,0,1]])
supercell.set_cell(supercell.cell*mask)


# NPT dynamics: Langevin Barostat for pressure control
dyn = NPT(supercell, timestep=time_step, temperature_K=initial_temp, 
          externalstress=pressure, ttime=100 * units.fs, pfactor=0.1*bulk_mod*75**2 * units.fs**2)
# Save trajectory
traj = Trajectory("LaMnO3_npt.traj", "w", supercell)
dyn.attach(traj.write, interval=10)  # Save every 100 steps

# Logging
dyn.attach(MDLogger(dyn, supercell, "npt.log", header=True, stress=True, peratom=True), interval=100)

# Function to manage temperature
def ramp_temperature(dynamics):
    if dynamics.nsteps>temp_ramp_steps:
        temp=final_temp
    else:
        temp=initial_temp
    dynamics.set_temperature(temperature_K=temp)
    print(f"Step {dynamics.nsteps}: Setting temperature to {temp:.2f} K")

# Attach temperature function
dyn.attach(ramp_temperature, interval=100, dynamics=dyn)
# Run simulation
dyn.run(n_steps)
    
