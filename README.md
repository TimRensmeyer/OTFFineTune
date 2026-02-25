This is a repository for finetuning foundational neural network potentials on the fly.
The fine-tuning mechanism is summarized in our paper "On-The-Fly Fine-Tuning of Foundational Neural Network Potentials: A Bayesian Neural Network Approach".

Because of dependency conflicts of different foundation models, the pyproject.toml requires a specification for which foundation model the dependencies should be set up.
To set it up for the MACE model, you have to run the command

pip install -e ".[MACE]"

For the NequIP model you have to run 

pip install -e ".[NequIP]"

In the Tests directory, unit tests are provided to assert that this module was set up correctly.
Further, an integration test is provided that runs a mock otf fine-tuning run. This test runs a little mock simulation of a hydrogen molecule with a toy potential energy model instead of VASP. Note, that the goal of this test is to assert that the workflow and process communication works as it should. Only two optimization steps are performed for each Monte Carlo sample at each update to keep compute times low, so no attempt is made to fit the labeled data well. This test is designed to be computationally light enough, that is can be run on the login nodes of a hpc or on a local workstation without any access to a gpu. The tests should be run from within the test directories itself (or with the individual test directories set as the working directory).

The dependencies of the pyproject.toml where tested by installing this repository in a fresh conda environment and passed all tests without errors. In case a dependency related issue happens anyway on other devices, we provided the full conda environments used for each model in the making of the paper within the Tutorials directories. There we also provide a guide for how to integrate VASP into on-the-fly simulations.

This current version of the repository is meant as a checkpointed version of the repository to be published on Zenodo. This code is a cleaned up but functionally and feature equivalent version of the code used the experiments in the paper. Future versions will be expanded by many features but might deviate in their workflow (e.g. sampling methods/ calibration procedure) to some degree. Therefore, to reproduce the experiments from the paper, this is the version you should use.

A high level overview for how this code works is given in the Documentation directory.

Future Plans:

I have been mainly busy on cleaning up the main branch of this repository and making it easier to use and install up to now.

However, for some side project application several features have been developed in forks or the dev branch of the repository. These features include

- gpu parallel inference capability during production runs
- using force uncertainties as an additional intervention criterion exist as features
- native ONETEP support
- native support of any electronic structure method that is set up as an ase calculator
- running VASP and the ML models as a SLURM HetJob on different nodes respectively.

Some optimization and testing still needs to be done for those features. However, after archiving this version of the repository on Zenodo for reproducibility and publishing of the paper, these features will be merged in the near future.

Further, I expect to a 4th generation neural network potential such as CHGNET to enable finetuning on systems with charge transfer.

It should be fairly straightforward to adapt this repository to work with other electronic structure methods by adding a custom interfarce for the electronic structure method to the Proc.py file in the procs/comm directory of the source code. The exact changes needed are documented in this file.
