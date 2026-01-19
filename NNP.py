import abc
from abc import abstractmethod
import os
import torch
import time
import torch.nn as nn
from typing import List
import numpy as np
import copy
from multiprocessing import Process, Queue
import multiprocessing as mp
import pickle

# Force spawn context to ensure clean CUDA initialization in child processes
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set in this context

import yaml
with open('runconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)

ErrorThreshold=config['ErrorThreshold']
try:
    ForceErrorThresholds=config['ForceErrorThresholds']
except:
    ForceErrorThresholds=None


from TrainProc import TrainProcComSetUp,SetTrainRequest,GetTrainStatus,SetTrainProcStatus
from LogPriors import GaussianMeanField
from MCMC import CyclicOptimizer
import subprocess


class NNP(abc.ABC,nn.Module):

    __metaclass__=abc.ABCMeta

    def __init__(self):
        super(NNP,self).__init__()

    #Input:
        #ase_struct: An ase structure for which the NNP is supposed to make a predictions
    #Output:
        # The output is expected to be a list of length at least 4 [E_pred,F_pred,E_std,F_std,...]
        # The frist 4 entries are pytorch tensors of shapes (1,),(N,3),(1,),(N,3) where N is the 
        # number of atoms in the structure.
    @abc.abstractmethod
    def predict(self,ase_atoms):
        ...

    #This method takes a new labeled data point and updates the NNP
    @abc.abstractmethod
    def update(self,new_data):
         ...
         




# A wrapper function to make predictions from an ensemble of models
# by fitting a Gaussian to the ensemble predictive distribution
def Gaussian_NNP_Ens(model_list,ase_atoms):

    Energies=[]
    Forces=[]
    Stress=[]
    E_stds=[]
    F_stds=[]
    Stress_stds=[]
    for m in model_list:
            out=m.predict(ase_atoms)
            if len(out)==6:
                StressIncluded=True
                (E,F,S,E_std,F_std,S_std)=out
                Stress.append(S.detach())
                Stress_stds.append(S_std.detach())
            else:
                StressIncluded=False
                (E,F,E_std,F_std)=out

            print(E.detach().cpu().item(),E_std.detach().cpu().item())
            Energies.append(E.detach())
            Forces.append(F.detach())
            E_stds.append(E_std.detach())
            F_stds.append(F_std.detach())

    Energies=torch.stack(Energies,dim=0)
    Forces=torch.stack(Forces,dim=0)
    E_stds=torch.stack(E_stds,dim=0)
    F_stds=torch.stack(F_stds,dim=0)

    E_var1=torch.var(Energies,dim=0)
    E_var2=torch.mean(E_stds**2,dim=0)
    F_var1=torch.var(Forces,dim=0)
    F_var2=torch.mean(F_stds**2,dim=0)
    
    E_std=((E_var1+E_var2)**0.5).detach().cpu().numpy()
    F_std=((F_var1+F_var2)**0.5).detach().cpu().numpy()
    E=torch.mean(Energies,dim=0).detach().cpu().numpy()
    F=torch.mean(Forces,dim=0).detach().cpu().numpy()

    if StressIncluded:
        Stress=torch.stack(Stress,dim=0)
        S_std=torch.stack(Stress_stds,dim=0)
        S_var1=torch.var(Stress,dim=0)
        S_var2=torch.mean(S_std**2,dim=0)
        S_std=((S_var1+S_var2)**0.5).detach().cpu().numpy()
        S=torch.mean(Stress,dim=0).detach().cpu().numpy()
        return [E,F,S,E_std,F_std,S_std]
    else:
        return [E,F,E_std,F_std]



def _inference_worker(worker_id, model_dict, device_id, input_queue, output_queue):
    """
    Worker process that runs inference on a model.
    
    Reconstructs model in worker process with proper device initialization.
    Receives ASE atoms from input_queue, runs prediction, sends results to output_queue.
    Terminates when receiving None.
    """
    try:
        # Reconstruct model in worker process with proper device
        model = model_dict['model']
        device = torch.device(f"cuda:{device_id}")
        model = model.to(device)
        
        while True:
            ase_atoms = input_queue.get()
            if ase_atoms is None:  # Shutdown signal
                break
            try:
                out = model.predict(ase_atoms)
                # Detach all tensors in output to allow pickling across process boundary
                detached_out = []
                for tensor in out:
                    if isinstance(tensor, torch.Tensor):
                        detached_out.append(tensor.detach())
                    else:
                        detached_out.append(tensor)
                output_queue.put((worker_id, detached_out, None))
            except Exception as e:
                output_queue.put((worker_id, None, str(e)))
    except Exception as e:
        output_queue.put((worker_id, None, f"Worker initialization failed: {str(e)}"))


def Gaussian_NNP_Ens_Par(process_list, input_queues, output_queues, ase_atoms):
    """
    Parallel wrapper function for ensemble predictions.
    
    Similar to Gaussian_NNP_Ens but uses multiprocessing:
    - Distributes ase_atoms to all worker processes via input_queues
    - Collects results from output_queues
    - Combines predictions using Gaussian ensemble logic
    - Handles tensors on different devices by moving to CPU before stacking
    
    Args:
        process_list: List of Process objects
        input_queues: List of Queue objects (input for each worker)
        output_queues: List of Queue objects (output from each worker)
        ase_atoms: ASE atoms object for prediction
        
    Returns:
        List containing [E, F, E_std, F_std] or [E, F, S, E_std, F_std, S_std]
    """
    # Send atoms to all workers
    for input_queue in input_queues:
        input_queue.put(ase_atoms)
    
    # Collect results from all workers
    Energies = []
    Forces = []
    Stress = []
    E_stds = []
    F_stds = []
    Stress_stds = []
    StressIncluded = False
    
    for i, output_queue in enumerate(output_queues):
        worker_id, out, error = output_queue.get()
        if error is not None:
            raise RuntimeError(f"Worker {worker_id} failed: {error}")
        
        if len(out) == 6:
            StressIncluded = True
            (E, F, S, E_std, F_std, S_std) = out
            # Move tensors to CPU to avoid cross-GPU memory issues
            Stress.append(S.cpu() if isinstance(S, torch.Tensor) else S)
            Stress_stds.append(S_std.cpu() if isinstance(S_std, torch.Tensor) else S_std)
        else:
            (E, F, E_std, F_std) = out
        
        # Move tensors to CPU before collecting them
        E_cpu = E.cpu() if isinstance(E, torch.Tensor) else E
        F_cpu = F.cpu() if isinstance(F, torch.Tensor) else F
        E_std_cpu = E_std.cpu() if isinstance(E_std, torch.Tensor) else E_std
        F_std_cpu = F_std.cpu() if isinstance(F_std, torch.Tensor) else F_std
        
        print(E_cpu.item() if isinstance(E_cpu, torch.Tensor) else E_cpu, 
              E_std_cpu.item() if isinstance(E_std_cpu, torch.Tensor) else E_std_cpu)
        
        Energies.append(E_cpu)
        Forces.append(F_cpu)
        E_stds.append(E_std_cpu)
        F_stds.append(F_std_cpu)
    
    # Combine predictions using Gaussian logic (all tensors now on CPU)
    Energies = torch.stack(Energies, dim=0)
    Forces = torch.stack(Forces, dim=0)
    E_stds = torch.stack(E_stds, dim=0)
    F_stds = torch.stack(F_stds, dim=0)
    
    E_var1 = torch.var(Energies, dim=0)
    E_var2 = torch.mean(E_stds ** 2, dim=0)
    F_var1 = torch.var(Forces, dim=0)
    F_var2 = torch.mean(F_stds ** 2, dim=0)
    
    E_std = ((E_var1 + E_var2) ** 0.5).detach().cpu().numpy()
    F_std = ((F_var1 + F_var2) ** 0.5).detach().cpu().numpy()
    E = torch.mean(Energies, dim=0).detach().cpu().numpy()
    F = torch.mean(Forces, dim=0).detach().cpu().numpy()
    
    if StressIncluded:
        Stress = torch.stack(Stress, dim=0)
        S_std = torch.stack(Stress_stds, dim=0)
        S_var1 = torch.var(Stress, dim=0)
        S_var2 = torch.mean(S_std ** 2, dim=0)
        S_std = ((S_var1 + S_var2) ** 0.5).detach().cpu().numpy()
        S = torch.mean(Stress, dim=0).detach().cpu().numpy()
        return [E, F, S, E_std, F_std, S_std]
    else:
        return [E, F, E_std, F_std]



class EnsembleFF(nn.Module):
          
     def __init__(self, device_list,n_models, constructor,constructor_args,restart=False,path=''):
          self.model_list=[]
          self.dev_models=[[] for dev in device_list]
          if constructor=='SpiceNequIP':
              from SpiceModelLoader import NequIP_Loader,NequIP_Wrapper,NequIP_Builder
              builder=NequIP_Builder
          elif constructor=='MACE':
              from MACE_Loader import MACE_Builder
              builder=MACE_Builder
          else:
              print('Error: Model constructor {} not recognized'.format(constructor))
          for i in range(n_models):
               m=builder(constructor_args)
               dev=i%len(device_list)
               self.dev_models[dev].append(i)
               self.model_list.append(m)
          
          self.dev_models=[models for models in self.dev_models if models!=[]]

          self.device_list=device_list
          pred_dev=torch.device("cuda:{}".format(device_list[0]))
          self.pred_dev=pred_dev
          self.model_list=[m.to(pred_dev) for m in self.model_list]
          self.nprocs=len(self.dev_models)
          self.path=path
          print('procs:',self.nprocs,self.dev_models)
          TrainProcComSetUp(self.nprocs)

          #Starting up training processes
          for proc_number in range(self.nprocs):
              n_models=len(self.dev_models[proc_number])
              dev=self.device_list[proc_number] 
              arg_list= ['{}'.format(arg) for arg in constructor_args]
              init_type='I'
              if restart:
                  init_type='R'

              model_count=len(self.dev_models[proc_number]) 
              SetTrainProcStatus(proc_number,'Starting Up')   
              OTF_dir = os.path.dirname(os.path.realpath(__file__))
              command=["python3","-u",OTF_dir+"/Training.py",'{}'.format(proc_number),
                       '{}'.format(dev),'{}'.format(n_models),constructor,init_type,self.path] +arg_list
              subprocess.Popen(command,stdout=open("tmp/training{}.log".format(proc_number), "w"))
          if restart:
            #loading model states
            i=0
            for proc_number in range(self.nprocs):
                for model_id in range(len(self.dev_models[proc_number])):                    
                    model=self.model_list[i]
                    model=model.to(torch.device('cpu'))   # This may look stupid but the memories of the GPUs in our hpc arent linked properly so we have to take a cpu detour.
                    model.load_state_dict(torch.load('model_dict{}{}'.format(proc_number,model_id),map_location=torch.device('cpu')))
                    model=model.to(self.pred_dev)
                    self.model_list[i]=model
                    i+=1

     def shutdown(self):
         for proc_number in range(self.nprocs):
             SetTrainProcStatus(proc_number,'Shutdown')
         
     def predict(self,ase_atoms):
          
          return Gaussian_NNP_Ens(self.model_list, ase_atoms)
     
     def update(self,new_data):
          
          torch.save(new_data,'tmp/new_data')
          SetTrainRequest(self.nprocs)
          done=False
          while not done:
            time.sleep(0.1)
            status=GetTrainStatus(self.nprocs)
            if status=='Finished':
                print('Update completed')
                done=True

        #loading updated models
          i=0
          for proc_number in range(self.nprocs):
              for model_id in range(len(self.dev_models[proc_number])):
                  model=self.model_list[i]
                  model=model.to(torch.device('cpu')) # This may look stupid but the memories of the GPUs in our hpc arent linked properly so we have to take a cpu detour.
                  model.load_state_dict(torch.load('model_dict{}{}'.format(proc_number,model_id),map_location=torch.device('cpu')))
                  model=model.to(self.pred_dev)
                  self.model_list[i]=model
                  i+=1
                

class EnsembleFFPar(nn.Module):
     """
     Parallel version of EnsembleFF that runs inference in separate processes.
     
     Instead of sequential inference through model_list, this class:
     - Maintains a list of worker processes, each with an input/output queue pair
     - Each process runs one model and performs predictions in parallel
     - Models are updated by terminating and relaunching processes with new weights
     """
     #ToDo: Move model creations to worker processes for parallel startup + pickle avoidance
     def __init__(self, device_list, n_models, constructor, constructor_args, restart=False, path=''):
          super(EnsembleFFPar, self).__init__()
          self.model_list = []
          self.dev_models = [[] for dev in device_list]
          
          if constructor == 'SpiceNequIP':
              from SpiceModelLoader import NequIP_Loader, NequIP_Wrapper, NequIP_Builder
              self.builder = NequIP_Builder
          elif constructor == 'MACE':
              from MACE_Loader import MACE_Builder
              self.builder = MACE_Builder
          else:
              print('Error: Model constructor {} not recognized'.format(constructor))
          
          # Build initial models
          for i in range(n_models):
               m = self.builder(constructor_args)
               dev = i % len(device_list)
               self.dev_models[dev].append(i)
               self.model_list.append(m)
          
          self.dev_models = [models for models in self.dev_models if models != []]
          
          self.device_list = device_list
         # pred_dev = torch.device("cuda:{}".format(device_list[0]))
         # self.pred_dev = pred_dev
         # self.model_list = [m.to(pred_dev) for m in self.model_list]
          self.nmodels = len(self.model_list)
          self.constructor = constructor
          self.constructor_args = constructor_args
          self.path = path
          self.restart = restart
          
          # For training process management (kept for compatibility with training infrastructure)
          self.nprocs = len(self.dev_models)
          print('parallel inference procs:', self.nprocs, self.dev_models)
          TrainProcComSetUp(self.nprocs)
          

          
          # Start training processes if needed
          for proc_number in range(self.nprocs):
              n_models = len(self.dev_models[proc_number])
              dev = self.device_list[proc_number]
              arg_list = ['{}'.format(arg) for arg in constructor_args]
              init_type = 'I'
              if restart:
                  init_type = 'R'
              
              model_count = len(self.dev_models[proc_number])
              SetTrainProcStatus(proc_number, 'Starting Up')
              OTF_dir = os.path.dirname(os.path.realpath(__file__))
              command = ["python3", "-u", OTF_dir+"/Training.py", '{}'.format(proc_number),
                         '{}'.format(dev), '{}'.format(n_models), constructor, init_type, self.path] + arg_list
              subprocess.Popen(command, stdout=open("tmp/training{}.log".format(proc_number), "w"))
          
          if restart:
              # Loading model states
              i = 0
              for proc_number in range(self.nprocs):
                   for model_id in range(len(self.dev_models[proc_number])):
                        model = self.model_list[i]
                        model = model.to(torch.device('cpu'))
                        model.load_state_dict(torch.load('model_dict{}{}'.format(proc_number, model_id), map_location=torch.device('cpu')))
                       # model = model.to(self.pred_dev)
                        self.model_list[i] = model
                        i += 1
          # Initialize inference processes
          self._init_inference_processes()

     def _init_inference_processes(self):
          """
          Initialize inference worker processes.
          Each worker process gets one model and a pair of input/output queues.
          Models are moved to CPU before passing to ensure proper pickling.
          """
          self.input_queues = []
          self.output_queues = []
          self.processes = []
          
          for i, model in enumerate(self.model_list):
               input_queue = Queue()
               output_queue = Queue()
               
               # Move model to CPU for pickling
               model_cpu = model.to(torch.device('cpu'))
               model_dict = {'model': model_cpu}
               device_id = self.device_list[i % len(self.device_list)]
               
               # Create and start worker process with proper device handling
               p = Process(target=_inference_worker, args=(i, model_dict, device_id, input_queue, output_queue))
               p.daemon = False  # Explicit lifecycle management
               p.start()
               
               self.input_queues.append(input_queue)
               self.output_queues.append(output_queue)
               self.processes.append(p)
     
     def _restart_inference_processes(self):
          """
          Terminate current inference processes and spawn new ones with updated models.
          This is called when models are updated.
          """
          # Signal shutdown to all workers
          for input_queue in self.input_queues:
               input_queue.put(None)
          
          # Wait for processes to finish
          for p in self.processes:
               p.join(timeout=5)
               if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                         p.kill()
          
          # Clear old resources
          self.input_queues.clear()
          self.output_queues.clear()
          self.processes.clear()
          
          # Reinitialize with updated models
          self._init_inference_processes()
     
     def shutdown(self):
          """Cleanly shutdown all inference processes and training processes."""
          # Shutdown inference processes
          for input_queue in self.input_queues:
               input_queue.put(None)
          
          for p in self.processes:
               p.join(timeout=5)
               if p.is_alive():
                    p.terminate()
          
          # Shutdown training processes
          for proc_number in range(self.nprocs):
               SetTrainProcStatus(proc_number, 'Shutdown')
     
     def predict(self, ase_atoms):
          """
          Perform parallel inference using worker processes.
          
          Returns list with predictions combined using Gaussian ensemble logic.
          """
          return Gaussian_NNP_Ens_Par(self.processes, self.input_queues, self.output_queues, ase_atoms)
     
     def update(self, new_data):
          """
          Update models with new training data.
          
          Steps:
          1. Save new data
          2. Signal training processes
          3. Wait for training to complete
          4. Load updated models
          5. Restart inference processes with new models
          """
          torch.save(new_data, 'tmp/new_data')
          SetTrainRequest(self.nprocs)
          done = False
          while not done:
               time.sleep(0.1)
               status = GetTrainStatus(self.nprocs)
               if status == 'Finished':
                    print('Update completed')
                    done = True
          
          # Load updated models
          i = 0
          for proc_number in range(self.nprocs):
               for model_id in range(len(self.dev_models[proc_number])):
                    model = self.model_list[i]
                    model = model.to(torch.device('cpu'))
                    model.load_state_dict(torch.load('model_dict{}{}'.format(proc_number, model_id), map_location=torch.device('cpu')))
                  #  model = model.to(self.pred_dev)
                    self.model_list[i] = model
                    i += 1
          
          # Restart inference processes with updated models
          self._restart_inference_processes()

                
from Procs import FileIOReqHandler
import ase
import scipy

def Confidence(e_bound,std,n,E,a,b):
    E_eff=0.5*E+b
    d=(n+1)/2+a
    denom=E_eff**0.5*2**0.5*std
    gam_log_num=scipy.special.gammaln(d)
    gam_log_denom=scipy.special.gammaln(d-0.5)
    prefactor=(2*np.pi*std**2)**(-0.5)
    Z=prefactor*np.exp(gam_log_num-gam_log_denom)
    Z*=2/(E_eff**0.5)
    conf=scipy.special.hyp2f1(0.5,d,1.5,-(e_bound/denom)**2)*e_bound

    return conf*Z

def ForceConfidence(F_bounds,stds,n,F,a,b):
    n_atoms=stds.shape[0]
    if len(F_bounds)==1:
        F_bounds=np.array([[F_bounds]*3]*n_atoms)
    else:
        F_bounds=np.stack([F_bounds]*3,axis=1)

    if n<2000:
        print("Shape Check:", F_bounds.shape,stds.shape)
        F_eff=0.5*F+b
        d=(n+1)/2+a
        denom=F_eff**0.5*2**0.5*stds
        gam_log_num=scipy.special.gammaln(d)
        gam_log_denom=scipy.special.gammaln(d-0.5)
        prefactor=(2*np.pi*stds**2)**(-0.5)
        Z=prefactor*np.exp(gam_log_num-gam_log_denom)
        Z*=2/(F_eff**0.5)
        conf=scipy.special.hyp2f1(0.5,d,1.5,-(F_bounds/denom)**2)*F_bounds

        return conf*Z
    else:
        Mn=F/n
        var_eff=stds**2*Mn
        conf=scipy.special.erf(F_bounds/(2*var_eff)**0.5)
        return conf

class OTFForceField(nn.Module):
    def __init__(self,MLFF,DFTReqHandler,E_thresh=ErrorThreshold,conf_thresh=0.95,restart=False):
        super(OTFForceField,self).__init__()
        self.MLFF=MLFF

        self.DFTReqHandler=FileIOReqHandler

        self.E_thresh=E_thresh
        self.F_Thresh=ForceErrorThresholds
        if self.F_Thresh != None:
            self.E_F=0
            self.n_F=0

        self.conf_thresh=conf_thresh
        self.FirstForward=True
        self.E=0
        self.n=0
        self.E_offset=0
        self.steps=0
        self.StressIncluded=True
        if restart:
            OTFParams=torch.load('tmp/OTFParams')
            self.E=OTFParams[0]
            self.n=OTFParams[1]
            self.E_offset=OTFParams[2]
            self.steps=OTFParams[3]
            self.FirstForward=False
            if self.F_Thresh!=None:
                self.E_F=OTFParams[4]
                self.n_F=OTFParams[5]
    


    def forward(self,atoms,log=True):
        self.steps+=1
        if isinstance(atoms, str):
            atoms=ase.io.read(atoms)
        preds=self.MLFF.predict(atoms)
        if len(preds)==6:
            self.StressIncluded=True
            [E_pred,F_pred,S_pred,E_uncert,F_uncert,S_uncert]=preds

        else:
            self.StressIncluded=False
            [E_pred,F_pred,E_uncert,F_uncert]=preds

        conf=Confidence(self.E_thresh,E_uncert,self.n,self.E,a=1.5,b=10)
        if self.F_Thresh != None:
            F_conf=ForceConfidence(self.F_Thresh,F_uncert,self.n_F,self.E_F,a=1.5,b=10)
            F_conf_min=np.min(F_conf)
        else:
            print("no Force threshold")
            F_conf=1
            F_conf_min=1

        if log:
            preds.append(conf)
            preds.append(F_conf)
            torch.save(preds,'ML_preds/{}'.format(self.steps))
            torch.save(atoms,'Coords/{}'.format(self.steps))

        print("Predicted Energy:",E_pred)
        print("Predicted Confidence:", conf,F_conf_min)
        print("Confidence Arguments:",self.E_thresh,E_uncert,self.n,self.E)
        print("Atom Types:", atoms.get_atomic_numbers())
        if self.F_Thresh != None:
            print("Confidence Arguments:",self.n_F,self.E_F,np.mean(F_uncert**2)**0.5)
        if (conf<self.conf_thresh or F_conf_min<self.conf_thresh or self.steps==1):
            t0= time.time()
            dft_out=self.DFTReqHandler(atoms)
            t1=time.time()
            print("DFT Calculation took:",t1-t0,"seconds")
            
            if dft_out[-1].any() == 'FAILED':
                atoms,E,F,S,tmp=dft_out
            elif len(dft_out)==4:
                atoms,E,F,S=dft_out
                E+=self.E_offset
                self.update([atoms,E,F,S],[conf,F_conf])
            else:
                atoms,E,F=dft_out
                E+=self.E_offset
                self.update([atoms,E,F],[conf,F_conf])
            t2=time.time()
            print("Update took:",t2-t1,"seconds")
            if self.FirstForward:
                E+=self.E_offset
            self.FirstForward=False
            if log:
                DFT_pred=(E,F)
                torch.save(DFT_pred,'DFT_preds/{}'.format(self.steps))
            if len(dft_out)==4:
                return (atoms,E,F,S,E*0,F*0,S*0)
            else:
                return (atoms,E,F,E*0,F*0)
        
        else:
            
            return [atoms]+preds[:-2]
        
    def recalibrate(self,new_data,confidences):
        E_conf,F_conf=confidences
        if len(new_data)==4:
            atoms,E,F,S=new_data

        else:
            atoms,E,F=new_data

        out = self.MLFF.predict(atoms)
        if len(out)==4:
            (E_pred,F_pred,E_uncert,F_uncert)=out
        else:
            (E_pred,F_pred,S_pred,E_uncert,F_uncert,S_uncert)=out
        print('Force Error:',F_pred-F)
        if self.FirstForward:
            self.E_offset=E_pred-E
        # Updating calibration of energy and force uncertainty based
        # on all predictions where the probability of error exceeding
        # the threshold was at least 75% of the maximum allowed probability
        elif self.steps>2:
            
            if (1-E_conf)>(1-self.conf_thresh)*0.75:
                print("Recalibrating Energy Error Model")
                self.E+=((E-E_pred)**2/E_uncert**2)
                self.n+=1

            if self.F_Thresh != None:
                if np.min(F_conf)<self.F_Thresh:
                    print("Recalibrating Force Error Model")
                    recalibration_mask=((1-F_conf)>(1-self.F_Thresh)*0.75)
                    #shape of forces is (N,3) where N is the number of atoms
                    #gathering all indices of force components that need recalibration:

                    indices=np.nonzero(recalibration_mask.astype(int))
                    F_uncert_masked=F_uncert[indices]
                    F_masked=F[indices]
                    F_pred_masked=F_pred[indices]
                    print("recalibration check:",F_masked.shape,F_pred_masked.shape,F_uncert_masked.shape)
                    self.E_F+=np.sum((F_masked-F_pred_masked)**2/F_uncert_masked**2)
                    self.n_F+=len(F_masked)

    


    def update(self,new_data,confidences):
        
        self.recalibrate(new_data,confidences)
        if self.FirstForward:
            new_data[1]+=self.E_offset
        if len(new_data)==4 and not self.StressIncluded:
            new_data=new_data[:3]
        self.MLFF.update(new_data)
        if self.F_Thresh == None:
            OTFParams=(self.E,self.n,self.E_offset,self.steps)
        else:
            OTFParams=(self.E,self.n,self.E_offset,self.steps,self.E_F,self.n_F)
        torch.save(OTFParams,'tmp/OTFParams')
          



