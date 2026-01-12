from Procs import SetProcStatus, GetProcStatus
import time
import subprocess
import os

import yaml
import sys

proc_id = int(os.environ.get("SLURM_PROCID", 0))
srcpath=sys.argv[1]
trgpath=sys.argv[2]
#sys.path.insert(0, srcpath)

if __name__ == "__main__":
    with open('runconfig.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    if 'DFTReferenceSource' in config.keys():
        DFTReqHandler = config['DFTReferenceSource']
    else:
        # VASP + SLURM is set as the default for now
        DFTReqHandler = 'VASPSLURM'

    SetProcStatus("Job Running")

    if DFTReqHandler == 'VASPSLURM':
        run_command = ["srun","--chdir="+trgpath,'vasp_std'] # Change TargetDir to Run Directory.
    elif DFTReqHandler == 'ONETEPSLURM':
        run_command=['python3','onetep.py']


    done=False
    while not done:
        status=GetProcStatus()
        if status =='DFT Request' and proc_id==0:
            print(trgpath)
            SetProcStatus('DFT Calculating')
            print(run_command)
            # Run the command and wait for it to finish
           # result = subprocess.run(["vasp_std"], cwd=trgpath, check=True)
            #run_vasp(trgpath, nodes=2)
            #srun_command = ["srun", "--chdir="+trgpath, "vasp_std"]
            #print(srun_command)
            # Run the command and wait for it to finish
            # subprocess.run(srun_command, check=True)
           # SetProcStatus('Finished Calculating')
            # result = subprocess.run(run_command, check=True)

            # check set to false for now because onetep seems to trigger a non-zero exit
            result = subprocess.run(run_command, check=False)
            print(result)
            SetProcStatus('Finished Calculating')

        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)