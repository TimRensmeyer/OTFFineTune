"""Mock VASP process for testing purposes. Instead of running VASP calculations,
it just sleeps for a short time and then updates the status to Finished."""

import time 
import sys

srcpath=sys.argv[1]
trgpath=sys.argv[2]
sys.path.insert(0, srcpath)

from OTFFineTune.Procs import SetProcStatus, GetProcStatus


if __name__ == "__main__":
    SetProcStatus("Job Running")
    print(trgpath)

    

    done=False
    while not done:
        status=GetProcStatus()
        if status =='DFT Request':
            SetProcStatus('DFT Calculating')
            # Run the command and wait for it to finish
            time.sleep(2) # Simulate time taken for DFT calculation
            SetProcStatus('Finished Calculating')

        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)