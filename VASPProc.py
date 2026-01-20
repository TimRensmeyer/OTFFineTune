"""
VASP DFT Calculation Subprocess

This module runs as a separate process and handles DFT calculations via VASP.
It waits for geometry input from the main process, runs VASP calculations,
and signals when results are ready for retrieval.

Communication:
- Monitors tmp/status.txt for 'DFT Request' signal
- Reads POSCAR from working directory
- Runs VASP via srun (for HPC clusters)
- Signals 'Finished Calculating' when done
- Main process reads OUTCAR for energies and forces

The subprocess runs in the target simulation directory (TargetPath) and uses
standard VASP input files (INCAR, KPOINTS, POSCAR).

Entry Point:
    python VASPProc.py <code_path> <target_path>
    
    Args:
        code_path: Path to code repository
        target_path: Working directory for VASP calculations
"""

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
            result = subprocess.run(srun_command, check=True)
            SetProcStatus('Finished Calculating')

        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)