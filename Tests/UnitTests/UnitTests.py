'''This file implements the actual tests. As testing becomes more comprehensive, 
this file will be split into multiple test files.
The tests cover various components of the OTF fine-tuning workflow, including model construction,
 optimizer functionality, process communication, and end-to-end workflow testing with mock DFT request handlers.
The test configuration is loaded from a YAML file, allowing for flexible specification of parameters such as the model
 type (MACE or NequIP) and the source directory for the code.'''

import os
import time
import yaml
import ase

with open('runconfig.yaml', 'r') as f:
    conf = yaml.safe_load(f)
#This file is located in CodePath/OTFFineTune/Tests/UnitTests, so we need to go up 
# four levels to get to the CodePath
src_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))   
NNP=conf['NNPBuilder']
import sys
sys.path.insert(0, src_dir)
#Because the test is not in the same directory as the source files, 
# we need to add the source directory to the python path to import the necessary modules for testing.
sys.path.insert(0, os.path.join(src_dir,"OTFFineTune/"))
from OTFFineTune.src.OTFFineTune.procs.comm.Procs import GetGPUProcStatus, SetGPUProcStatus
from OTFFineTune.Tests.Utils import MockDFTReqHandler, MockDFTReqHandlerNoStress

def test_model_construction_and_forward_pass() -> None:
    """Test the construction of the Network class and a forward pass with dummy input data.
    This test validates that the model can be instantiated and that the forward method produces outputs of the expected shape."""
    if NNP=='MACE':
        from OTFFineTune.src.OTFFineTune.models.MACE_Loader import MACE_Builder
        builder_func=MACE_Builder
    elif NNP=='SpiceNequIP':
        from OTFFineTune.src.OTFFineTune.models.SpiceModelLoader import NequIP_Builder
        builder_func=NequIP_Builder
    else:
        raise ValueError("Invalid model type specified in test configuration. Expected 'MACE' or 'NequIP'.")
    
    #Construct model with dummy parameters (these would need to be consistent with the actual model builders)
    model = builder_func([0.5])  # Example parameters, adjust as needed for actual builders
    #create and ase atoms H2 dimer for testing
    from ase import Atoms
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])  # H-H bond length ~0.

    if NNP=='MACE':
        atoms.set_cell([10, 10, 10])  # Set a large cell to avoid interactions with periodic images
        atoms.set_pbc([True, True, True])  
        energy, forces, stress, e_unc, f_unc, s_unc = model.predict(atoms)  # Forward pass
        assert energy.shape == (1,)  # Check energy output shape
        assert forces.shape == (2, 3)  # Check forces output shape for 2 atoms
        assert stress.shape == (1, 3, 3)   # Check stress output shape and if 

    elif NNP=='SpiceNequIP':

        energy, forces, e_unc, f_unc = model.predict(atoms)  # Forward pass
        assert energy.shape == (1,1)  # Check energy output shape
        assert forces.shape == (2, 3)  # Check forces output shape

    print("Model construction and forward pass test passed.")

def test_optimizer_step() -> None:
    """Test the optimizer step functionality of the CyclicOptimizer class.
    This test validates that the optimizer can perform a step and that the model parameters are updated accordingly."""
    from OTFFineTune.src.OTFFineTune.core.MCMC import CyclicOptimizer, GaussianMeanField
    from OTFFineTune.Tests.Utils import DummyDataLoader, DummyModelConstructor
    import torch
    #Construct dummy model and data loader for testing
    #setting up means and variances for the dummy GaussianMeanField distribution used in the optimizer test
    
    model = DummyModelConstructor()
    means = []
    stds = []
    for param in model.parameters():
        means.append(torch.zeros_like(param))
        stds.append(torch.ones_like(param))
    distribution = GaussianMeanField(means, stds)
    data_loader = DummyDataLoader()
    optimizer = CyclicOptimizer(model,distribution, data_loader, cycle_length=10,max_lr=0.1)  # Use a single cycle for testing
    print("Entering optimizer tests. Log Likelihoods should improve throughout" \
    " the optimization cycle.")
    optimizer.run(model)  # Perform a short optimization cylce
    print("Optimizer run test completed. Check log likelihoods for improvement.")

def test_process_communication() -> None:
    """Tests if the Process status files read and write work as expected."""
    from OTFFineTune.src.OTFFineTune.procs.comm.Procs import SetProcStatus, GetProcStatus,  SetUp
    from OTFFineTune.src.OTFFineTune.procs.comm.TrainProc import TrainProcComSetUp, SetTrainRequest, GetTrainStatus, SetTrainProcStatus

    #remove tmo folder if it exists
    import shutil
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    
    SetUp()
    TrainProcComSetUp(3)
    time.sleep(0.1)  # Allow some time for setup to complete
    SetTrainRequest(3)
    time.sleep(0.1)  # Allow some time for status updates to propagate
    status=GetTrainStatus(3)
    assert status=='running', f"Expected 'Running' but got '{status}'"
    for i in range(3):
        SetTrainProcStatus(i,'Finished')
    time.sleep(0.1)  # Allow some time for status updates to propagate
    status=GetTrainStatus(3)
    assert status=='Finished', f"Expected 'Finished' but got '{status}'"
    print("Training process communication test passed.")
    SetProcStatus('Finished Calculating')
    status=GetProcStatus()
    assert status=='Finished Calculating', f"Expected 'Finished Calculating' but got '{status}'"
    print("Process communication test passed.")

    SetGPUProcStatus('Finished OTF Calculation')
    status=GetGPUProcStatus()
    assert status=='Finished OTF Calculation', f"Expected 'Finished OTF Calculation' but got '{status}'"
    print("GPU Process communication test passed.")

