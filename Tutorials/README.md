This folder contains some examples for how to set up simulations with the source code.

Each example folder contains a conda environment yaml to checkpoints the exact python dependencies from the paper.
In practice it is easier to set this repository up with the pyproject.toml, so if you used it to 
install this repository into any environment manager, then you dont need to reinstall it here. However, the exact dependency versions might differ from the ones used for the paper in this case.

Specific changes that need to made on the user side to set up the simulations:

-  I provided INCAR, POSCAR and KPOINTS files for VASP but for licensing reasons, I can't provide the POTCAR files. So a suitable file needs to be set up.
-  The SLURM script "MLFFProc_Submit" has to be adjusted to the specific HPC cluster. Also the conda environment name should match the name of the actual environment with the correct dependencies set up. The exact lines that need to be adjusted are marked in the file itself with elaborations.

If this is done, the simulations can simply be run with 

python -u run.py

Note that it is expected that this command is run from within the individual Tutorial Example directories themself or with those directories set as the working directory. E.g. running 

python -u CaZrS3Example/run.py 

from this directory will not work correctly.

Additionally, there is a runconfig.yaml for setting up the machine learning specific aspects of the run.
Nothing in those files needs to be changed for these tutorials to run but you can play around with some
settings if you want.
