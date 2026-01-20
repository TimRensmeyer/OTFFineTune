"""
Inter-Process Communication and DFT Interface

This module handles communication between main process and GPU/DFT subprocesses
through temporary file-based status updates. Provides utilities for:
- Initialization of working directories and status files
- Status synchronization via temporary files
- DFT request handling (VASP interface)
- Process launching with SLURM support

Note: File-based communication is used instead of queues for HPC compatibility
and checkpoint/restart robustness. Temporary files are stored in ./tmp/

 Currently, only a VASP interface exists. However, to add an interface for a different
electronic structure method, only a few changes are needed. This interface should mirror the FileIOReqHandlerVASP()
function in this file. 
Then two small further changes are needed in the OTFForceField class in NNP.py and MLFFProc.py.
The exact changes needed are documented in the corresponding files
"""

from torch.multiprocessing import Pool
import subprocess
import torch
import torch.nn as nn
import numpy as np
import os
import time
import ase
from ase.io import read,write

# Utilities for main process-to-subprocess communication
def ProcComSetUp():
    """Initialize main process communication infrastructure."""
    os.mkdir('tmp')
    fp=open('./tmp/status.txt', 'w')
    fp.write('ready')
    fp.close()

def SetUp():
    """Initialize complete working directory structure for OTF simulation."""
    dircont=os.listdir('./')

    for dn in ['Coords','ML_preds','DFT_preds','Checkpoints']:
        if dn not in dircont:
            os.mkdir(dn)
        
    if 'tmp' not in dircont:
        ProcComSetUp()
        fp=open('./tmp/gpu_status.txt', 'w')
        fp.write('initialized')
        fp.close()


def SetProcStatus(status):
    fp=open('./tmp/status.txt', 'w')
    fp.write(status)
    fp.close()

def GetProcStatus():
    fp=open('./tmp/status.txt', 'r')
    status=fp.read()
    fp.close()
    return status

def GPUProcComSetUp():
    os.mkdir('tmp')
    fp=open('./tmp/gpu_status.txt', 'w')
    fp.write('ready')
    fp.close()

def SetGPUProcStatus(status):
    fp=open('./tmp/gpu_status.txt', 'w')
    fp.write(status)
    fp.close()

def GetGPUProcStatus():
    fp=open('./tmp/gpu_status.txt', 'r')
    status=fp.read()
    fp.close()
    return status


def ProcLauncher(SLURMFILE=None,PROCFILE=None,Restart=False):
    if Restart:
        SetGPUProcStatus('Restart')
    else:
        SetGPUProcStatus('Initiating')

    # If a SLURMFILE was specified, the proc gets launched via SLURM
    if SLURMFILE!=None:
        os.popen('sbatch ' + SLURMFILE )

    elif PROCFILE!=None:
        proc = subprocess.Popen(['python3', PROCFILE])

def FileIOReqHandlerVASP(atoms):
    """
    VASP DFT Request Handler.
    
    Manages the complete VASP DFT calculation workflow:
    1. Write atomic structure to POSCAR
    2. Signal DFT process to start calculation
    3. Wait for completion
    4. Extract energies, forces (and stresses) from OUTCAR
    
    Energies and forces are converted from eV to eV/Angstrom for consistency.
    Stress is converted from kB to eV.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        - If stress available: (atoms, energy, forces, stress)
        - Otherwise: (atoms, energy, forces)

    To adapt this code for other electronic structure codes,
    a similar interface should be implemented. The requirements are, that it takes
    an ase atoms structure as input and returns the atoms object as well as the
    energy, forces, and optionally stresses. All outputs should use kcal/mol as
    energy units and Angstrom as length units.
    """

    
    # Forwarding Request to VASPProc
    SetProcStatus('DFT Request')

    #Waiting for the Calculation to finish
    while GetProcStatus() != 'Finished Calculating':
        time.sleep(1)

    # extracting data from outcar
    atoms_out=read("OUTCAR", index=':')[0]
    energy=atoms_out.get_potential_energy()*23.0609
    forces=atoms_out.get_forces()*23.0609
    try:
        [xx,yy,zz,yz,zx,xy]=list(atoms_out.get_stress())
        stress=np.array([[xx,xy,zx],
                        [xy,yy,yz],
                        [zx,yz,zz]])*23.0609
        return atoms,energy,forces,stress
    except:       
        return atoms,energy,forces

def FileIOReqHandlerOTF(atoms,IncludeStress=False):
    """
    OTF GPU Process Request Handler.
    
    Submits inference request to GPU subprocess and retrieves results:
    1. Write atoms to temporary XYZ file
    2. Signal GPU process for OTF prediction
    3. Wait for prediction completion
    4. Load numpy-serialized predictions and uncertainties
    
    Args:
        atoms: ASE Atoms object
        IncludeStress: Whether to retrieve stress predictions
        
    Returns:
        - If IncludeStress: (atoms, energy, forces, stress, e_uncert, f_uncert, s_uncert)
        - Otherwise: (atoms, energy, forces, e_uncert, f_uncert)
    """
    write('tmp/atoms.xyz', atoms)

    # Forwarding Request to VASPProc
    SetGPUProcStatus('OTF Request')

    #Waiting for the Calculation to finish
    while GetGPUProcStatus() != 'Finished OTF Calculation':
        time.sleep(0.01)

    atoms=read('tmp/atoms.xyz')
    forces=np.load('tmp/forces.npy')
    energy=np.load('tmp/energy.npy')
    e_uncert=np.load('tmp/e_uncert.npy')
    f_uncert=np.load('tmp/f_uncert.npy')

    if IncludeStress:
        stress=np.load('tmp/stress.npy')
        s_uncert=np.load('tmp/s_uncert.npy')
        return atoms,energy,forces,stress,e_uncert,f_uncert,s_uncert
    else:
        return atoms,energy,forces,e_uncert,f_uncert

def VASPSLURMBuilder(SLURMFILE):
    """
    Build VASP DFT request handler with SLURM job submission.
    
    Note: This function is currently no longer used and may be removed in future versions.
    Consider using FileIOReqHandlerVASP directly.
    """

    return FileIOReqHandlerVASP()

def OTFSlurmBuilder(SLURMFILE):
    """
    Build OTF GPU process launcher with SLURM job submission.
    
    Note: This function is currently no longer used and may be removed in future versions.
    Consider using ProcLauncher directly.
    """


    return FileIOReqHandlerOTF

def SlurmStartup(
                 OTFBUILDER=OTFSlurmBuilder,
                 GPUSLURMFILE="MLFFProc_Submit",restart=False):
    """
    Startup OTF workflow with SLURM resource management.
    
    Launches both DFT (CPU) and GPU (ML) processes via SLURM, waiting for both
    to become ready before returning the inference request handler.
    
    Note: This function is currently no longer used and may be removed in future versions.
    The startup workflow has been integrated into MLFFProc.py.
    
    Args:
        OTFBUILDER: Function to build GPU process launcher
        GPUSLURMFILE: Path to GPU process SLURM submission script
        restart: Whether to restore from checkpoint
        
    Returns:
        OTF request handler callable
    """
    if 'tmp' not in os.listdir('./'):
        SetUp()
        
    if restart:
        SetGPUProcStatus("Restart")

    OTFReqHandler=OTFBUILDER(GPUSLURMFILE)

    launched=False
    while not launched:
        gpu_status=GetGPUProcStatus()
        gpu_launched=(gpu_status=="OTF Force Field Ready")
        cpu_status=GetProcStatus()
        cpu_launched=(cpu_status=="Job Running")
        if cpu_launched and gpu_launched:
            launched=True
            break
    return OTFReqHandler
