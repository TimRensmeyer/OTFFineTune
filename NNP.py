import abc
from abc import abstractmethod
import torch
import time
import torch.nn as nn
from typing import List
import numpy as np
import copy

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
              command=["python3","-u","/beegfs/home/r/rensmeyt/Git/OTFFineTune/Training.py",'{}'.format(proc_number),
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
                
from Procs import FileIOReqHandlerVASP
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

class OTFForceField(nn.Module):
    def __init__(self,MLFF,DFTReqHandler,E_thresh=ErrorThreshold,conf_thresh=0.95,restart=False):
        super(OTFForceField,self).__init__()
        self.MLFF=MLFF
        if DFTReqHandler=='VASPSLURM':
            self.DFTReqHandler=FileIOReqHandlerVASP
        else:
            self.DFTReqHandler=DFTReqHandler
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
            F_conf=np.min(F_conf)
        else:
            print("no Force threshold")
            F_conf=1

        if log:
            preds.append(conf)
            torch.save(preds,'ML_preds/{}'.format(self.steps))
            torch.save(atoms,'Coords/{}'.format(self.steps))

        print("Predicted Energy:",E_pred)
        print("Predicted Confidence:", conf)
        print("Confidence Arguments:",self.E_thresh,E_uncert,self.n,self.E)
        print("Atom Types:", atoms.get_atomic_numbers())
        if self.F_Thresh != None:
            print("Confidence Arguments:",self.n_F,self.E_F,np.mean(F_uncert**2)**0.5)
        if (conf<self.conf_thresh or F_conf<self.conf_thresh or self.steps==1):

            dft_out=self.DFTReqHandler(atoms)
            if len(dft_out)==4:
                atoms,E,F,S=dft_out
                E+=self.E_offset
                self.update([atoms,E,F,S])
            else:
                atoms,E,F=dft_out
                E+=self.E_offset
                self.update([atoms,E,F])

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
            
            return [atoms]+preds[:-1]
        
    def recalibrate(self,new_data):
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

        elif self.steps>2:
      
            self.E+=(E-E_pred)**2/E_uncert**2
            self.n+=1
            if self.F_Thresh != None:
                print("calibration check:",((F-F_pred)**2).shape,F_uncert.shape)
                self.E_F+=np.sum((F-F_pred)**2/F_uncert**2)
                self.n_F+=np.sum((F-F_pred)*0+1)
    


    def update(self,new_data):
        self.recalibrate(new_data)
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
          



