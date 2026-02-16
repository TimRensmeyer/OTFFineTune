'''In this module unit and integration tests for the OTF fine-tuning workflow are implemented.
 These tests cover:
 - Model constructors and forward passes for the Network class
 - Testing of the optimizer step and run methods for the CyclicOptimizer class
 - Process communication 
 - End-to-end testing of the MLFFProc workflow with a mock DFT request handler and a simple test potential model
 
 Note: 
        - The model type (MACE or NequIP) is determined by the "NNP" variable in the test configuration file "test_config.yaml".
        - These tests are designed to run in a CPU environment and do not require GPU resources.
            They are intended to validate the core functionality of the code and the process communication logic, rather than
            the performance of the models.'''