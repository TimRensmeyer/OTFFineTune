This is a repository for finetuning foundational neural network potentials on the fly.
The fine-tuning mechanism is summarized in our paper "On-The-Fly Fine-Tuning of Foundational Neural Network Potentials: A Bayesian Neural Network Approach".

Disclaimer:
 The current version of the code is what could be generously called "research code".
 Meaning its still a mess.
 It currently requires adjusting some settings directly in python files.
 Also I have two versions of the code. One for the NequIP model and one for the MACE model because adding stresses and periodic boundaries broke some sections of the initial code which I developed for the NequIP model. I can send the other version of the code to anyone brave enough to try to use this version of the code but this will be fixed soon anyway.

Future Plans:
 I am currently working on cleaning this code base and making it user friendly.
 I plan on adding examples of how to use this codebase.
 One limitation is that I can only test and develop this code for SLURM managed HPC clusters since this is what I have available. I will try to make the code clean enough so that adapting it to other settings will be easy.

 Also high on my list is adding force uncertainties as an additional intervention criterion, so that for specific atoms or regions of interest lower uncertainty thresholds can be chosen than for others.
 Further I expect to add ONETEP support in the near future as well as a 4th generation neural network potential such as CHGNET to enable finetuning on larger systems and systems with charge transfer.
