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
import NNP
import ase
import yaml
import numpy as np
#import shutup
import sys



if __name__ == "__main__":
   # shutup.please()
    with open('runconfig.yaml', 'r') as file:
        config = yaml.safe_load(file)

    CodePath=config['CodePath']
    TargetPath=config['TargetPath']
    sys.path.insert(0, CodePath)

    from OTFFineTune.Procs import SetGPUProcStatus, GetGPUProcStatus,GPUProcComSetUp,SetProcStatus
    from OTFFineTune.TrainProc import GetTrainStatus

    Restart=(GetGPUProcStatus()=="Restart")
    #You can remove the next two lines if you use a different electronic structure method.
    #  VASPProc.py is just used as part of the default vasp interface function.
    command="python3 "+CodePath+"OTFFineTune/VASPProc.py"+ " " +CodePath +" "+TargetPath
    os.popen(command)
    SetGPUProcStatus("OTF Force Field Starting Up")
    done=False


  
    
    n_procs=len(config['dev_list'])
    MLFF=NNP.EnsembleFF(device_list=config['dev_list'],
                        n_models=config['n_models'], 
                        constructor=config['NNPBuilder'],
                        constructor_args=config['constructor_args'],restart=Restart,path=CodePath)
    
    OTFForceField=NNP.OTFForceField(MLFF=MLFF,
                                    DFTReqHandler='VASPSLURM',#You may want to change this if you implemented a custom electronic structure inteface
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
            done=True
            break
        else:
            time.sleep(0.01)
