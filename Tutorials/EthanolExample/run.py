import os 
import sys

sys.path.insert(0, '/beegfs/home/r/rensmeyt/Git/')

print('test1')
for fn in os.listdir():
    if fn[-4:]=='.out':
        os.remove(fn)
print('test2')

import yaml

from OTFFineTune.Procs import SlurmStartup

Handler=SlurmStartup(restart=False)

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import ase
    
class OTF_Calcualator(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, req_handler):
        super().__init__()
        self.req_handler = req_handler

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        atoms,energy,forces,E_uncert,F_uncert= self.req_handler(atoms,IncludeStress=False)
        # Store the results in the `_results` dictionary
        self.results['energy'] = energy/23.0609
        self.results['forces'] = forces/23.0609




calc=OTF_Calcualator(Handler)
from ase.build import molecule
atoms = molecule('CH3CH2OH')
if not any(atoms.pbc):
    atoms.cell=[[30,0,0],[0,30,0],[0,0,30]]
    atoms.pbc=True
atoms.calc = calc



from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 0.5 * units.fs,temperature_K=300,friction=0.01 / ase.units.fs,trajectory='md.traj', logfile='md.log')
dyn.run(200)

