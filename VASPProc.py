from Procs import SetProcStatus, GetProcStatus
import time
import subprocess
import os

import yaml
import sys
srcpath=sys.argv[1]
trgpath=sys.argv[2]
#sys.path.insert(0, srcpath)




if __name__ == "__main__":
    SetProcStatus("Job Running")
    print(trgpath)
    srun_command = ["srun","--chdir="+trgpath, "vasp_std" ] # Change TargetDir to Run Directory.
    

    done=False
    while not done:
        status=GetProcStatus()
        if status =='DFT Request':
            SetProcStatus('DFT Calculating')
            # Run the command and wait for it to finish
           # result = subprocess.run(srun_command, check=True)
            SetProcStatus('Finished Calculating')

        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)