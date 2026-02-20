This folder contains some examples for how to set up simulations with the source code.

Each example folder contains a conda environment yaml to set up the python dependencies for that example.
These are equivalent to the environments you can install with the pyproject.toml, so if you used pip to 
install this repository into any environment manager, then you dont need to reinstall it here. 
Just make sure that in the MLFFProcSubmit script you adjust how you load the environment.

Additionally, there is a runconfig.yaml for setting up the machine learning specific aspects of the run.
Nothing in those files needs to be changed for these tutorials to run but you can play around with some
settings if you want.

Specific changes that need to made on the user side to set up the simulations:

-  I provided INCAR, POSCAR and KPOINTS files for VASP but for licensing reasons, I can't provide the POTCAR files. So a suitable file needs to be set up.
-  The SLURM script "MLFFProc_Submit" has to be adjusted to the specific HPC cluster. Also the conda environment name should match the name of the actual environment with the correct dependencies set up. Lastly the paths have to be adjusted to match your system.

If this is done, the simulations can simply be run with 

python -u run.py