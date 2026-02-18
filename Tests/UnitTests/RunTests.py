'''In this file, we run all the tests defined in Tests.py.
 The test configuration is read from "test_config.yaml" in the Tests directory,
   which specifies the model type (MACE or NequIP) to be used in the tests.'''


import UnitTests
import ase
import numpy as np
import torch
import yaml
import os
if __name__ == "__main__":
  #setting random seeds for reproducibility of the tests
  torch.manual_seed(0)
  np.random.seed(0)

  with open('runconfig.yaml', 'r') as f:
      conf = yaml.safe_load(f)
  src_dir=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
  NNP=conf['NNPBuilder']
  import sys
  sys.path.insert(0, src_dir)
 # from OTFFineTune.Tests.Utils import MockDFTReqHandler, MockDFTReqHandlerNoStress, LoadTestConfig
  UnitTests.test_model_construction_and_forward_pass()
  UnitTests.test_optimizer_step()
  UnitTests.test_process_communication()

  #removing temporary files created during testing
  import shutil
  import os
  temp_folders = ['Checkpoints', 'Coords', 'DFT_preds', 'ML_preds', 'tmp']

  for folder in temp_folders:
      if os.path.exists(folder):
          shutil.rmtree(folder)