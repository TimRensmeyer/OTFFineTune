"""A mock MLFF process for testing purposes. Orchestrates the 
mock training and DFT request handling processes. No actual models are built or trained,
 and no real DFT calculations are performed."""

import time
import subprocess
import os
from MLFFProc import E_pred
import NNP
from Procs import FileIOReqHandlerVASP
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
    command="python3 "+CodePath+"OTFFineTune/Tests/MockVASPProc.py"+ " " +CodePath +" "+TargetPath
    os.popen(command)
    SetGPUProcStatus("OTF Force Field Starting Up")
    done=False


  
    
    n_procs=3
    for proc_number in range(n_procs):
        dev=config['dev_list'][proc_number]
        constructor=config['NNPBuilder']
        init_type=config['init_type']
        arg_list=config['constructor_args']
        command=["python3","-u",CodePath+"OTFFineTune/Tests/MockTraining.py",'{}'.format(proc_number),
                    '{}'.format(dev),'{}'.format(3),constructor,init_type,CodePath] +arg_list
        subprocess.Popen(command,stdout=open("tmp/training{}.log".format(proc_number), "w"))
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
            # Calling Mock DFT calculation
            out=FileIOReqHandlerVASP(atoms)

            np.save('tmp/energy.npy',out[0])
            np.save('tmp/forces.npy',out[1])

            SetGPUProcStatus('Finished OTF Calculation')

        elif status=='Shutdown':
            SetProcStatus(status)
            done=True
            break
        else:
            time.sleep(0.01)
