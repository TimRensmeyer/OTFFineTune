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

    for dn in ['Coords','ML_preds','DFT_preds','Checkpoints','SimFiles']:
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

def FileIOReqHandler(atoms):

    n_current_step = len(os.listdir('Coords'))

    # # Pure MACE results for training - debug option
    # import yaml
    # with open('runconfig.yaml', 'r') as file:
    #     config = yaml.safe_load(file)
    # if 'DFTReferenceSource' in config.keys() and config['DFTReferenceSource'] == 'MACE':
    #     if len(os.listdir('SimFiles')) == 0:
    #         print('running MACE')
    #         from mace.calculators import mace_mp
    #         calc = mace_mp(model="medium",dispersion=False,default_dtype="float32",device='cpu',return_raw_model=False)
    #         atoms.calc = calc
    #         energy=atoms.get_potential_energy()*23.0609
    #         forces=atoms.get_forces()*23.0609
    #         [xx,yy,zz,yz,zx,xy]=list(atoms.get_stress())
    #         stress=np.array([[xx,xy,zx],
    #                         [xy,yy,yz],
    #                         [zx,yz,zz]])*23.0609
    #     else:
    #         print('running MLFF')
    #         atoms,energy,forces,stress,e_uncert,f_uncert,s_uncert = FileIOReqHandlerOTF(atoms,IncludeStress=True)
    #     write('SimFiles/MLFF{}.xyz'.format(n_current_step),atoms,'extxyz')
    #     return atoms,energy,forces,stress

    write('POSCAR',atoms,'vasp')
    
    # Forwarding Request to DFTProc
    SetProcStatus('DFT Request')

    #Waiting for the Calculation to finish
    while GetProcStatus() != 'Finished Calculating':
        time.sleep(1)

    # extracting data from outcar
    DFT_succeeded = True

    out_format = ''
    if os.path.isfile('OUTCAR'):
        out_file = 'OUTCAR'
        out_format = 'vasp-out'
    elif os.path.isfile('onetep.out'):
        out_file = 'onetep.out'
        out_format = 'onetep-out'
    elif os.path.isfile('espresso.pwo'):
        out_file = 'espresso.pwo'
        out_format = 'espresso-out'
    else:
        DFT_succeeded = False
    

    try:
        atoms_out=read(out_file, index=-1, format=out_format)
        energy=atoms_out.get_potential_energy()*23.0609
        forces=atoms_out.get_forces()*23.0609
        text=os.popen('cp {} SimFiles/{}{}'.format(out_file,out_file,n_current_step)).read()
    except: 
        DFT_succeeded = False
    os.remove(out_file)

    if not DFT_succeeded:
        print('DFT calculation failed at step {}. Working with MLFF prediction instead.'.format(n_current_step))
        return 'DFT FAILED'
    else:
        print(text)

    try:
        [xx,yy,zz,yz,zx,xy]=list(atoms_out.get_stress())
        stress=np.array([[xx,xy,zx],
                        [xy,yy,yz],
                        [zx,yz,zz]])*23.0609
        return atoms,energy,forces,stress
    except:       
        return atoms,energy,forces

def FileIOReqHandlerOTF(atoms,IncludeStress=False):
    with open('tmp/atoms.xyz', 'w') as f:
        write(f, atoms)
    print('write finished')
    print(f"[DEBUG] MLFFProc PID: {os.getpid()}, Host: {os.uname()[1]}")
    #time.sleep(1)
    # Forwarding Request to DFTProc
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
    
class OTFReqHandler():
    def __init__(self,proc):
        self.proc=proc
    def __call__(self,atoms,IncludeStress=False):
        return FileIOReqHandlerOTF(atoms,IncludeStress)
    def shutdown(self):
        if self.proc is not None:
            SetGPUProcStatus('Shutdown')
            SetProcStatus('Shutdown')
            return_text=os.popen('scancel '+ self.proc).read()

            #self.proc.join()
        return True


def OTFSlurmBuilder(SLURMFILE):

    if SLURMFILE == 'proc':
        OTF_dir = os.path.dirname(os.path.realpath(__file__))
        out = subprocess.Popen(['python3', '-u', OTF_dir+'/MLFFProc.py'])
        out = 0
    else:
        out=os.popen('sbatch '+ SLURMFILE).read()
        out=out.split(' ')[-1]
        out=out[:-1]
        print("Job Id:",out)
    req_handler=OTFReqHandler(out)

    return req_handler

def SlurmStartup(
                 OTFBUILDER=OTFSlurmBuilder,
                 GPUSLURMFILE="MLFFProc_Submit",restart=False):
    
    #if 'tmp' not in os.listdir('./'):
    SetUp()
        
    if restart:
        print('Doing restart')
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
