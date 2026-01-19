'''
Main program for running and coordinating anything related to DFT and MLFF
'''


import time
import subprocess
import os
import ase
import yaml
import numpy as np
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
    import OTFFineTune.NNP as NNP

    Restart=(GetGPUProcStatus()=="Restart")
    # If the resources are allocated via a SLURM hetjob, the DFT calculations (default: VASP) need to 
    # be launched in the slurm script directly. Otherwise the are launched automtically
    # from DFTProc.py
    if 'HetJob' not in config.keys():
        command="python3 "+CodePath+"OTFFineTune/DFTProc.py"+ " " +CodePath +" "+TargetPath
        os.popen(command)

    elif not config['HetJob']:
        command="python3 "+CodePath+"OTFFineTune/DFTProc.py"+ " " +CodePath +" "+TargetPath
        os.popen(command)
    

    SetGPUProcStatus("OTF Force Field Starting Up")
    done=False

    n_procs=len(config['dev_list'])
    #testchange
    MLFF=NNP.EnsembleFFPar(device_list=config['dev_list'],
                        n_models=config['n_models'], 
                        constructor=config['NNPBuilder'],
                        constructor_args=config['constructor_args'],restart=Restart,path=CodePath)
   
    # if 'DFTReferenceSource' in config.keys():
    #     DFTReqHandler = config['DFTReferenceSource']
    # else:
    #     DFTReqHandler = 'VASPSLURM'
 
    OTFForceField=NNP.OTFForceField(MLFF=MLFF,
                                    DFTReqHandler='VASPSLURM',
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
            print('OTF Calculating')
            t0=time.time()
            atoms=ase.io.read('tmp/atoms.xyz')
            t1=time.time()
            # Run the command and wait for it to finish
            out=OTFForceField(atoms)
            t2=time.time()
            print(len(out))
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
            t3=time.time()
            print(f"Time to read atoms: {t1-t0:.2f} s")
            print(f"Time for OTF Force Field Prediction: {t2-t1:.2f} s")
            print(f"Time to save results: {t3-t2:.2f} s")
            SetGPUProcStatus('Finished OTF Calculation')

        elif status=='Shutdown':
            SetProcStatus(status)
            done=True
            break
        else:
            time.sleep(0.1)
