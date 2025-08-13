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
    SetProcStatus("Job Running")

    srun_command = ["srun","--chdir="+trgpath, "vasp_std" ] # Change TargetDir to Run Directory.
    

    done=False
    while not done:
        status=GetProcStatus()
        if status =='DFT Request' and proc_id==0:
            print(trgpath)
            SetProcStatus('DFT Calculating')
            # Run the command and wait for it to finish
           # result = subprocess.run(["vasp_std"], cwd=trgpath, check=True)
            #run_vasp(trgpath, nodes=2)
            #srun_command = ["srun", "--chdir="+trgpath, "vasp_std"]
            #print(srun_command)
            # Run the command and wait for it to finish
            # subprocess.run(srun_command, check=True)
           # SetProcStatus('Finished Calculating')
            result = subprocess.run(srun_command, check=True)
            SetProcStatus('Finished Calculating')

        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)