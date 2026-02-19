'''This file contains test specific utility functions for the OTF fine-tuning workflow.
 These utilities include:
 - Mock DFT request handlers for testing process communication and data handling in the Procs module
 - Simple test potential model for end-to-end testing of the MLFFProc workflow
 - Configuration parsing for test cases
 - A dummy model constructor and data loader for testing the optimizers and training loop without
   relying on the actual MACE or NequIP model builders.
    '''

import numpy as np
import yaml
import sys
import os 
import ase
import torch
from typing import List, Union, Any
file_path=os.path.dirname(os.path.realpath(__file__))
parent_path=os.path.dirname(file_path)
sys.path.insert(0, parent_path)
sys.path.append("..")


def test_potential_model(atoms: ase.Atoms) -> List:
    """using a simple Lennard Jones potential as a test model for end-to-end testing of the MLFFProc workflow.
    Note that the Hydrogen dimer is used as a test system, so the potential is defined for H-H interactions.
    """
    #Lennard-Jones parameters for Hydrogen
    epsilon = 104  # Depth of the potential well in kcal/mol
    sigma = 0.661  # Finite distance at which the inter-particle potential is zero in Angstroms
    energy = 0.0
    forces = np.zeros((len(atoms), 3))
    positions = atoms.get_positions()
    distance_vector = positions[1] - positions[0]
    r = np.linalg.norm(distance_vector)

    # Calculate Lennard-Jones potential energy
    energy = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    # Calculate forces
    force_magnitude = 24 * epsilon * (2 * (sigma ** 12) / (r ** 13) - (sigma ** 6) / (r ** 7))
    forces[0] = -force_magnitude * (distance_vector / r)
    forces[1] = forces[0]  # Newton's third law
    return energy, forces # convert to kcal/mol and kcal/mol/Angstrom

def MockDFTReqHandler(atoms: ase.Atoms) -> List:
    import time
    from src.OTFFineTune.procs.comm.Procs import SetProcStatus, GetProcStatus

    """Mock DFT request handler that returns fixed energy and forces for testing.
    Follows the interface of the VASP DFT request handler used in the Procs module.
    Shapes of the returned energy, forces and stresses are consistent with typical DFT outputs
    for a system of the given atoms."""
    #Note: are the shapes of the numpy arrays correct?
    SetProcStatus('DFT Request')

    #Waiting for the Calculation to finish
    while GetProcStatus() != 'Finished Calculating':
        time.sleep(1)

    energy, forces = test_potential_model(atoms)
    [xx, yy, zz, yz, zx, xy] = [0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001]  # Fixed stress components
    stresses = np.array([[xx,xy,zx],
                        [xy,yy,yz],
                        [zx,yz,zz]])
    return atoms, energy, forces, stresses

def MockDFTReqHandlerNoStress(atoms: ase.Atoms) -> List:
    """Mock DFT request handler that returns fixed energy and forces without stress for testing.
    Follows the interface of the VASP DFT request handler used in the Procs module, but does not return stress."""
    energy, forces = test_potential_model(atoms)
    return atoms, energy, forces

#def LoadTestConfig(config_path):
 #   """Utility function to load test configuration from a YAML file.
  #  This allows test cases to specify parameters such as the model type (MACE or NequIP) and the source directory for the code.
   # The configuration is expected to be in a dictionary format with keys corresponding to the parameters used in the tests."""

   # with open(config_path, 'r') as file:
    #    config = yaml.safe_load(file)
    #return config
from src.OTFFineTune.core.MCMC import StochasticModel
def DummyModelConstructor() -> StochasticModel:
    """"constructs a small shallow fcnn with a single hidden layer and random weights. 
    This can be used for testing the optimizers and training loop without relying on 
    the actual MACE or NequIP model builders."""

    import torch.nn as nn

    class SimpleFCNN(StochasticModel):
        def __init__(self,
                     input_size: int = 3,
                     hidden_size: int = 5,
                     output_size: int = 1) -> None:
            super(SimpleFCNN, self).__init__()
            self.output_size=output_size
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size*2) #predictiong and std_dev

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            predictions = out[:, :self.output_size]
            std_devs = torch.exp(out[:, self.output_size:]) + 1e-3
            return predictions, std_devs
        
        def evaluate(self, data: List[torch.Tensor]) -> torch.Tensor:
            """Evaluation function expected by the CyclicOptimizer. 
            Computes a batch averaged log likelihood based on the predictions and targets."""
            inputs, targets = data
            predictions, std_devs = self.forward(inputs)
            # Compute log likelihood for each sample in the batch
            log_likelihoods = -0.5 * (torch.log(std_devs**2) + (targets - predictions)**2 / (std_devs**2))
            # Return average log likelihood over the batch
            print("Log likelihood:", log_likelihoods.mean().item())
            return log_likelihoods.mean()

    model = SimpleFCNN()
    return model

class DummyDataLoader:
    """A simple data loader that generates random input features and target values for testing the training loop and optimizers.
    Each batch contains random 3D coordinates as input features and scalar energies as target values."""

    def __init__(self,
                  batch_size: int = 5, 
                  num_batches: int = 1) -> None:

        self.bs = batch_size
        self.num_batches = num_batches
        self.inputs = [torch.randn(batch_size, 3) for _ in range(num_batches)]  # Random 3D coordinates
        self.targets = [torch.sum(inp, dim=1, keepdim=True) for inp in self.inputs]  # Simple target: sum of input features (just for testing)

    def sample(self) -> List[torch.Tensor]:
    #generate a random batch of data
        import random
        idx = random.randint(0, self.num_batches - 1)
        return self.inputs[idx], self.targets[idx]     

    
    def len(self) -> int:
        return self.bs * self.num_batches
