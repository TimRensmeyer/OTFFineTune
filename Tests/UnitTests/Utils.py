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
import os 
import ase
import torch
from typing import List, Union, Any



from OTFFineTune.core.MCMC import StochasticModel

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
