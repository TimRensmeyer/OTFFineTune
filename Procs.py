from torch.multiprocessing import Pool
import subprocess
import torch
import torch.nn as nn
import numpy as np
import os
import time
import ase
from ase.io import read,write

# Some basic utilies for communication between the subprocesses
def ProcComSetUp():
    os.mkdir('tmp')
    fp=open('./tmp/status.txt', 'w')
    fp.write('ready')
    fp.close()

def SetUp():
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

    # Generating VASP geometry 
    write('POSCAR',atoms,'vasp')
    
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

    os.popen('sbatch '+ SLURMFILE)

    return FileIOReqHandlerVASP()

def OTFSlurmBuilder(SLURMFILE):

    os.popen('sbatch '+ SLURMFILE) #testchange


    return FileIOReqHandlerOTF

def SlurmStartup(
                 OTFBUILDER=OTFSlurmBuilder,
                 GPUSLURMFILE="MLFFProc_Submit",restart=False):
    
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
