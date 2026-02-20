"""
ML Force Field GPU Subprocess

This module runs as a separate GPU process and handles on-the-fly force field
predictions with uncertainty quantification. It orchestrates the full OTF workflow:

1. Initialize ensemble of models across assigned GPUs
2. Start DFT subprocess (VASP)
3. Wait for training subprocesses to initialize
4. Poll for inference requests from main MD process
5. On request: predict energies/forces with uncertainties using ensemble
6. If confidence low: signal for DFT calculation and retrain
7. Return predictions for next MD step

Communication:
- Reads/writes tmp/atoms.xyz for structure exchange
- Uses tmp/gpu_status.txt for status synchronization
- Saves predictions as numpy arrays in tmp/

Entry Point:
    python MLFFProc.py
    
    Configuration (from runconfig.yaml):
        dev_list: List of GPU device IDs
        n_models: Number of models in ensemble
        NNPBuilder: Model type ('SpiceNequIP' or 'MACE')
        constructor_args: Arguments for model builder
        CodePath: Path to code repository
        TargetPath: Working directory for DFT calculations

To adapt the code for other electronic structure methods you can remove
the lines that launch the VASPProc.py process file. 
Further, if you chose to pass a keyword via the DFTReqHandler arg in the 
OTFForceField constructor to use the alternative electronic structure method,
then make sure that this keyword is passed when the constructor is called here.
"""

import time
import subprocess
import os

import ase
import yaml
import numpy as np
#import shutup
import sys



if __name__ == "__main__":
   # shutup.please()
    with open('runconfig.yaml', 'r') as file:
        config = yaml.safe_load(file)

    #If the CodePath and/or TargetPath are specified in the runconfig.yaml use those
    #Otherwise use the path to this file to determine the CodePath and
    #set the TargetPath as the working directory (i.e. the directory from which the simulation is launched)

    if 'CodePath' in config:

        CodePath=config['CodePath']
    else:
        #This file lives in CodePath/OTFFineTune/src/OTFFineTune/procs/MLFFProc.py, 
        # so we need to go up four levels to get to the CodePath
        file_path=os.path.dirname(os.path.realpath(__file__))
        CodePath=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
    if 'TargetPath' in config:
        TargetPath=config['TargetPath']
    else:
        TargetPath=os.getcwd()

    file_path=os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path)

    import OTFFineTune.core.NNP as NNP
    from comm.Procs import SetGPUProcStatus, GetGPUProcStatus,GPUProcComSetUp,SetProcStatus
    from comm.TrainProc import GetTrainStatus, SetTrainProcStatus

    Restart=(GetGPUProcStatus()=="Restart")

    #Check if the file is run inside the OTFFineTune/Tests/IntegrationTests folder.
    #  If so, launc the MockVASPProc instead of the actual VASPProc.
    #  This allows to run integration tests without access to VASP.
    # Further, in this case the ensemble force field will be launched with 
    # the testing=True flag which will limit the optimization to two steps
    # per optimization cycle and thus allow the test to run in a reasonable time frame.

    testing=False
    global_path=os.path.dirname(os.path.realpath(__file__))
    print(TargetPath)
    if 'OTFFineTune/Tests/IntegrationTest' in TargetPath:
        #For some reason, this does not work when the test is launched from the OTFFineTune/Tests/IntegrationTest folder
        testing=True
        print("Running in testing mode. Mock VASP process will be launched "
        "and optimization will be limited to two steps per cycle.")
    else:
        print("Running in normal mode. Actual VASP process will be launched "
        "and optimization will proceed without step limit.")
    #You can remove the next four lines if you use a different electronic structure method.
    #  VASPProc.py is just used as part of the default vasp interface function.
    
    #VASPProc.py is located in the same directory as this file, so the path is constructed accordingly.
    proc_path=os.path.dirname(os.path.realpath(__file__))
    if not testing:
        command="python3 "+proc_path+"/VASPProc.py"+ " " +CodePath +" "+TargetPath
        #command="python3 "+CodePath+"OTFFineTune/src/OTFFineTune/procs/VASPProc.py"+ " " +CodePath +" "+TargetPath
    else:
        command="python3 "+proc_path+"/MockVASPProc.py"+ " " +CodePath +" "+TargetPath

    os.popen(command)
    SetGPUProcStatus("OTF Force Field Starting Up")
    done=False


  
    
    n_procs=len(config['dev_list'])
    MLFF=NNP.EnsembleFF(device_list=config['dev_list'],
                        n_models=config['n_models'], 
                        constructor=config['NNPBuilder'],
                        constructor_args=config['constructor_args'],restart=Restart,path=CodePath,testing=testing)
    if not testing:
        OTFForceField=NNP.OTFForceField(MLFF=MLFF,
                                        DFTReqHandler='VASPSLURM',#You may want to change this if you implemented a custom electronic structure inteface
                                        restart=Restart)
    else:
        OTFForceField=NNP.OTFForceField(MLFF=MLFF,
                                        DFTReqHandler='Mock',#Use the mock DFT request handler for testing
                                        restart=Restart)
    ready=False
    while not ready:
        status=GetTrainStatus(n_procs)
        if status=='Finished':
            ready=True
            break
        time.sleep(0.1)
    
    SetGPUProcStatus("OTF Force Field Ready")
    while not done:
        status=GetGPUProcStatus()
        if status =='OTF Request':
            SetGPUProcStatus('OTF Calculating')
            atoms=ase.io.read('tmp/atoms.xyz')
            # Run the command and wait for it to finish
            out=OTFForceField(atoms)
            if len(out)==5:
                (atoms,E_pred,F_pred,E_uncert,F_uncert)=out
            else:
                (atoms,E_pred,F_pred,S_pred,E_uncert,F_uncert,S_uncert)=out
                np.save('tmp/stress.npy',S_pred)
                np.save('tmp/s_uncert.npy',S_uncert)

            np.save('tmp/energy.npy',E_pred)
            np.save('tmp/forces.npy',F_pred)
            np.save('tmp/e_uncert.npy',E_uncert)
            np.save('tmp/f_uncert.npy',F_uncert)

            SetGPUProcStatus('Finished OTF Calculation')

        elif status=='Shutdown':
            SetProcStatus(status)
            for i in range(n_procs):
                SetTrainProcStatus(i,'Shutdown')
            done=True
            break
        else:
            time.sleep(0.01)
