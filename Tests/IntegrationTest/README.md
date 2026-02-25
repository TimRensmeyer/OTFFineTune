This directory is used for integration testing of the code. 
By running 

python -u mockrun.py 

from within this directory, a complete mock simulation of a hydrogen molecule is run, except that instead of VASP a simple mock 
potential is used and only two optimization steps are used per Monte Carlo sample per update.
Therefore, the test is comutationally light enough that it can (and will) be run cpu resources to enable build-testing on hpc login nodes.
Because only two optimization steps are used each update, the models wont actually fit the mock potential well, so large errors
in the log files can be expected. 
This test is meant as a build-test and not as a reasonable simulation of a hydrogen molecule!

Note: You need to adjust the model loader in the runconfig.yaml to reflect the model setup you want to use (MACE or NequIP).
