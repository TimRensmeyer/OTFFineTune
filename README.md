This is a repository for finetuning foundational neural network potentials on the fly.
The fine-tuning mechanism is summarized in our paper "On-The-Fly Fine-Tuning of Foundational Neural Network Potentials: A Bayesian Neural Network Approach".

Disclaimer:
 The current version of the code is still in the transition period from messy research code to a code base that is easy to use for other users.

Future Plans:
 I am currently working on cleaning this code base and making it user friendly with more fleshed out tutorials on how to use it. I am planning to use the python Atomic Simulation Environment (ASE) package as the backbone for further developments, so that the on-the-fly force field will primarily be set up as an ASE calculator and any DFT software that is not natively supported but can be set up in ASE as a calculator can also easily be integrated into the on-the-fly force field.

 Also high on my list is adding force uncertainties as an additional intervention criterion, so that for specific atoms or regions of interest lower uncertainty thresholds can be chosen than for others.
 Further, I expect to add native ONETEP support in the near future as well as a 4th generation neural network potential such as CHGNET to enable finetuning on larger systems and systems with charge transfer.
