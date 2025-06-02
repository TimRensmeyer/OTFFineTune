from Procs import SetGPUProcStatus, GetGPUProcStatus,GPUProcComSetUp,SetProcStatus
from TrainProc import GetTrainStatus
import time
import subprocess
import os
import NNP
import ase
import yaml
import numpy as np
#import shutup

if __name__ == "__main__":
   # shutup.please()
    Restart=(GetGPUProcStatus()=="Restart")
    os.popen("python3 VASPProc.py")
    SetGPUProcStatus("OTF Force Field Starting Up")
    done=False


    with open('testconfig.yaml', 'r') as file:
        config = yaml.safe_load(file)
    n_procs=len(config['dev_list'])
    MLFF=NNP.EnsembleFF(device_list=config['dev_list'],
                        n_models=config['n_models'], 
                        constructor=config['NNPBuilder'],
                        constructor_args=config['constructor_args'],restart=Restart)
    OTFForceField=NNP.OTFForceField(MLFF=MLFF,
                                    DFTReqHandler='VASPSLURM',
                                    E_thresh=6,conf_thresh=0.95,restart=Restart)
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
