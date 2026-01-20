"""
OTFFineTune: On-the-Fly Fine-Tuning of Neural Network Potentials

This package implements an on-the-fly (OTF) fine-tuning algorithm for foundational
neural network potentials (NNPs). The system dynamically retrains ensemble models
during molecular dynamics simulations when prediction confidence is low, using
DFT calculations to generate high-quality training data adaptively.

Key Features:
- Ensemble-based uncertainty quantification for model confidence
- Parallel GPU-based inference and training processes
- Support for MACE and NequIP potential models
- MCMC-based optimization with transfer learning priors
- Weighted importance sampling for efficient data utilization
- VASP DFT integration for on-demand reference calculations

Main Components:
- NNP: Core force field classes (NNP, EnsembleFF, OTFForceField)
- MLFFProc: GPU process for model inference
- Training: Training subprocess launcher and manager
- Procs: Inter-process communication and DFT interface
- VASPProc: VASP DFT calculator wrapper
"""
