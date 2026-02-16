import os 
from pathlib import Path
import sys
import yaml
import time


print("Starting integration test for OTF fine-tuning workflow. This " \
    "can take a few minutes to run")

simulation_dir= Path(__file__).resolve().parent
runconfigfile=os.path.join(simulation_dir,'runconfig.yaml')

with open(runconfigfile, 'r') as f:
    runconfig = yaml.safe_load(f)

sys.path.insert(0, runconfig['CodePath'])
if runconfig['NNPBuilder']=='MACE':
    mode='MACE'
elif runconfig['NNPBuilder']=='SpiceNequIP':
    mode='NequIP'
else:
    raise ValueError("Invalid NNPBuilder specified in runconfig.yaml. Must be 'MACE' or 'NequIP'.")

for fn in os.listdir():
    if fn[-4:]=='.out':
        os.remove(fn)

from OTFFineTune.Procs import FileIOReqHandlerOTF, SetUp, GetProcStatus, GetGPUProcStatus
#instead of calling the OTFSlurmBuilder, we directly instantiate the request handler
# and launch the MLFFProc.py manually without SLURM for testing purposes. 
# This allows us to run the test on a single machine without requiring a SLURM cluster.
Handler=FileIOReqHandlerOTF
if 'tmp' not in os.listdir('./'):
    SetUp()


os.popen('python3 -u ' + os.path.join(runconfig['CodePath'],'OTFFineTune/MLFFProc.py >logs.txt'))

#Wait for the OTF process to be ready before proceeding with the test
launched=False
while not launched:
    gpu_status=GetGPUProcStatus()
    gpu_launched=(gpu_status=="OTF Force Field Ready")
    cpu_status=GetProcStatus()
    cpu_launched=(cpu_status=="Job Running")
    if cpu_launched and gpu_launched:
        launched=True
        break

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import ase
    
if mode=='MACE':

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
elif mode=='NequIP':

    class OTF_Calcualator(Calculator):
        
        implemented_properties = ['energy', 'forces']
        def __init__(self, req_handler):
            super().__init__()
            self.req_handler = req_handler

        def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)

            atoms,energy,forces,E_uncert,F_uncert = self.req_handler(atoms,IncludeStress=False)

            # Store the results in the `_results` dictionary
            self.results['energy'] = energy/23.0609
            self.results['forces'] = forces/23.0609

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
from ase.build import molecule

#create a hydrogen molecule with large pbc to test the OTF process
atoms=molecule('H2')
if not any(atoms.pbc):
    atoms.cell=[[30,0,0],[0,30,0],[0,0,30]]
    atoms.pbc=True
# Define simulation parameters
supercell=atoms
initial_temp =  500  # Start temperature
time_step = 1.5 * units.fs  # Time step in femtoseconds
n_steps = 20000  # Number of MD steps

# Set initial velocities
MaxwellBoltzmannDistribution(supercell, temperature_K=initial_temp)
#supercell.set_calculator(calc)  
supercell.calc=calc
import numpy as np
mask=np.array([[1,0,0],[0,1,0],[0,0,1]])
supercell.set_cell(supercell.cell*mask)
from ase.optimize import BFGS

dyn = Langevin(supercell, timestep=0.5*time_step,temperature_K=1500, friction=5e-1)
dyn.run(5)

#dyn=VelocityVerlet(supercell,timestep=time_step)

# Save trajectory
#traj = Trajectory("proton.traj", "w", supercell)
#dyn.attach(traj.write, interval=1)  # Save every 100 steps

# Logging
#dyn.attach(MDLogger(dyn, supercell, "npt.log", header=True, stress=True, peratom=True), interval=100)



# Run simulation
#dyn.run(n_steps)
 #shutdown the OTF process after the simulation is done
from OTFFineTune.Procs import SetGPUProcStatus

SetGPUProcStatus('Shutdown')

time.sleep(6)  # Give the OTF process time to shut down before the script exits
# After the simulation, remove the tmp directory and its contents
import shutil
#remove model dicts and npt.log

os.remove('logs.txt')
#if npt.log exists, remove it
if os.path.exists('npt.log'):
    os.remove('npt.log')
os.remove('model_dict00')
os.remove('model_dict10')
os.remove('model_dict20')
#remove the other folders and files created during the test
shutil.rmtree('tmp')
shutil.rmtree('Coords')
shutil.rmtree('ML_preds')
shutil.rmtree('DFT_preds')
shutil.rmtree('Checkpoints')
print("Integration test completed successfully.")



