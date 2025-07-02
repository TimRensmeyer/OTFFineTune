This folder contains some examples for how to set up simulations with the source code.
Note that this is a refactored version of the origina research code, that is not yet tested extensively. So there might still be some bugs left.
Each example folder contains a conda environment yaml to set up the python dependencies for that example.
Additionally, there is a runconfig.yaml for setting up the machine learning specific aspects of the run.

Specific changes that need to made on the user side to set up the simulations:

-  I provided INCAR, POSCAR and KPOINTS files for VASP but for licensing reasons, I can't provide the POTCAR files. So a suitable file needs to be set up.
-  The SLURM script "MLFFProc_Submit" has to be adjusted to the specific HPC cluster. Also the conda environment name should match the name of the actual environment with the correct dependencies set up. Lastly also the paths have to be adjusted
-  Two paths also have to be adjusted in the runconfig.yaml

If this is done, the simulations can simply be run with 

python -u run.py