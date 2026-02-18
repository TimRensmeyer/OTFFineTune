"""
Neural Network Potential (NNP) Framework

This module provides the core architecture for on-the-fly fine-tuning of neural network potentials.
It defines abstract base classes for NNP models and implements the ensemble-based uncertainty
quantification framework along with the main OTFForceField class that orchestrates the workflow.

Main Components:
- NNP: Abstract base class for neural network potentials
- Gaussian_NNP_Ens: Ensemble prediction with Gaussian posterior approximation
- EnsembleFF: Manages ensemble of models across multiple GPU processes
- OTFForceField: Main interface for on-the-fly fine-tuning orchestration
- Confidence: Calculates model confidence based on uncertainty estimates

The module integrates with subprocess-based training processes and supports
stress tensor predictions for enhanced structural accuracy.

To adapt this repository for an electronic structure software a small modification to the
OTFForceField class is needed and highlighted in the class implementation itself.
"""

import abc
from abc import abstractmethod
import torch
import time
import torch.nn as nn
from typing import List
import numpy as np
import copy
import os

import yaml

with open('runconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)

if 'CodePath' in config:
    CodePath=config['CodePath']
else:
    #This file lives in CodePath/OTFFineTune/src/OTFFineTune/procs/MLFFProc.py, 
    # so we need to go up four levels to get to the CodePath
    file_path=os.path.dirname(os.path.realpath(__file__))
    CodePath=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
if 'TargetPath' in config:
    TargetPath=config['TargetPath']
else:
    TargetPath=os.getcwd()
ErrorThreshold=config['ErrorThreshold']

from ..procs.comm.TrainProc import TrainProcComSetUp,SetTrainRequest,GetTrainStatus,SetTrainProcStatus
from .LogPriors import GaussianMeanField
from .MCMC import CyclicOptimizer
import subprocess


class NNP(abc.ABC,nn.Module):
    """
    Abstract base class for Neural Network Potential models.
    
    All NNP implementations should inherit from this class and provide:
    - predict(): Generate predictions (energies, forces, uncertainties) from atomic structures
    - update(): Add new labeled data and retrain the model
    """

    __metaclass__=abc.ABCMeta

    def __init__(self):
        super(NNP,self).__init__()

    @abc.abstractmethod
    def predict(self,ase_atoms):
        """
        Generate predictions for a given atomic structure.
        
        Args:
            ase_atoms: ASE Atoms object representing atomic structure
            
        Returns:
            List of predictions: [E_pred, F_pred, E_std, F_std, ...]
            - E_pred (Tensor): Predicted energy, shape (1,)
            - F_pred (Tensor): Predicted forces, shape (N,3) where N is number of atoms
            - E_std (Tensor): Energy uncertainty, shape (1,)
            - F_std (Tensor): Force uncertainty, shape (N,3)
            - Optional: S_pred, S_std (stress and stress uncertainty)
        """
        ...

    @abc.abstractmethod
    def update(self,new_data):
        """
        Update the NNP with new labeled data and retrain.
        
        Args:
            new_data: List containing [atoms, energy, forces] or [atoms, energy, forces, stress]
        """
        ...

         




# A wrapper function to make predictions from an ensemble of models
# by fitting a Gaussian to the ensemble predictive distribution
def Gaussian_NNP_Ens(model_list,ase_atoms):
    """
    Ensemble prediction with Gaussian posterior approximation.
    
    Combines predictions from multiple models by:
    1. Collecting individual predictions with aleatoric uncertainties
    2. Computing ensemble variance (epistemic uncertainty)
    3. Combining with mean aleatoric uncertainty
    4. Fitting a Gaussian to the predictive distribution
    
    Args:
        model_list: List of NNP models
        ase_atoms: ASE Atoms object
        
    Returns:
        List of predictions [E, F, S, E_std, F_std, S_std] or [E, F, E_std, F_std]
        containing posterior means and standard deviations combining:
        - Epistemic uncertainty (ensemble variance)
        - Aleatoric uncertainty (mean model confidence)
    """


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
    """
    Ensemble Force Field Manager.
    
    Manages an ensemble of neural network potentials distributed across multiple
    GPU processes. Handles:
    - Model initialization and distribution across devices
    - Training subprocess management (SGHMC optimization)
    - Synchronized model updates and checkpointing
    - Ensemble predictions with uncertainty quantification
    
    The class spawns independent training processes for each GPU device,
    communicating through shared temporary files. Model state is serialized
    for cross-process synchronization.
    
    Args:
        device_list: List of CUDA device IDs
        n_models: Total number of models in ensemble
        constructor: Model builder name ('SpiceNequIP' or 'MACE')
        constructor_args: Arguments for model builder
        restart: Whether to restore from checkpoint
        path: Code repository path
    """
     
    def __init__(self, device_list,n_models, constructor,constructor_args,restart=False,path='',testing=False):
        self.model_list=[]
        self.dev_models=[[] for dev in device_list]
        if constructor=='SpiceNequIP':
            from ..models.SpiceModelLoader import NequIP_Loader,NequIP_Wrapper,NequIP_Builder
            builder=NequIP_Builder
        elif constructor=='MACE':
            from ..models.MACE_Loader import MACE_Builder
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
        # For build testing, target_dev will be cpu to enable testing on login nodes without gpu access. 
        # In this case we set pred_dev to cpu as well.
        if testing:
            pred_dev=torch.device('cpu')
        else:
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
            file_dir = os.path.dirname(os.path.realpath(__file__))
            #OTF_dir is parent directory of the current file
            OTF_dir = os.path.dirname(file_dir)
            if not testing:
                command=["python3","-u",OTF_dir+"/procs/Training.py",'{}'.format(proc_number),
                       '{}'.format(dev),'{}'.format(n_models),constructor,init_type,self.path] +arg_list
            else:
                command=["python3","-u",OTF_dir+"/procs/TestTraining.py",'{}'.format(proc_number),
                       '{}'.format('cpu'),'{}'.format(n_models),constructor,init_type,self.path] +arg_list
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
        """Shutdown all training subprocesses gracefully."""
        for proc_number in range(self.nprocs):
            SetTrainProcStatus(proc_number,'Shutdown')
        
    def predict(self,ase_atoms):
        """
        Generate ensemble predictions with uncertainty quantification.
        
        Args:
            ase_atoms: ASE Atoms object
            
        Returns:
            Gaussian-approximated ensemble predictions
        """
        
        return Gaussian_NNP_Ens(self.model_list, ase_atoms)
    
    def update(self,new_data):
        """
        Add new labeled data and trigger ensemble retraining.
        
        Serializes data and signals all training processes to perform
        SGHMC optimization updates. Synchronizes until all models complete
        retraining before loading updated model states.
        
        Args:
            new_data: [atoms, energy, forces] or [atoms, energy, forces, stress]
        """

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
                
from ..procs.comm.Procs import FileIOReqHandlerVASP
import ase
import scipy

def Confidence(e_bound,std,n,E,a,b):
    """
    Calculate model confidence using Student's t-distribution.
    
    Uses a Gamma distribution based confidence metric to quantify
    reliability of energy predictions. Combines energy error bounds with
    uncertainty estimates and cumulative squared errors.
    
    Args:
        e_bound: Energy error threshold
        std: Energy standard deviation
        n: Number of accumulated squared errors
        E: Cumulative squared energy error
        a, b: Gamma distribution hyperparameters
        
    Returns:
        float: Confidence value in range [0, 1]
    """
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

class OTFForceField(nn.Module):
    """
    On-the-Fly Force Field with Active Learning.
    
    Main orchestrator for on-the-fly fine-tuning of neural network potentials.
    Manages the feedback loop between:
    - ML inference (EnsembleFF predictions with uncertainty)
    - Confidence assessment 
    - DFT reference calculations (when confidence < threshold)
    - Model retraining with new data
    
    The class maintains cumulative statistics for energy calibration and
    confidence scoring. Predictions are cached alongside DFT calculations
    for analysis and debugging.
    
    Args:
        MLFF: EnsembleFF instance
        DFTReqHandler: DFT calculator interface ('VASPSLURM' or custom callable)
        E_thresh: Energy error threshold for confidence calculation
        conf_thresh: Confidence threshold for triggering DFT calculation (default 0.95)
        restart: Whether to restore from checkpoint

    To adapt this repository for a different electronic structure method,
    it is required that the DFTReqHandler is set to the custom interface function 
    implemented in Procs.py. 
    This class is constructed in the MLFF.py file. The argument 'DFTReqHandler' is used
    to differentiate which electronic structure interface should be used.
    You can either chose to change the code here to ignore this argument and
    always use the desired electronic structure interface or you can add a new keyword 
    and condition to set this interface as the DFTReqHandler and make sure that in 
    MLFFProc.py the corresponding keyword is passed as an argument.
    """
    def __init__(self,MLFF,DFTReqHandler,E_thresh=ErrorThreshold,conf_thresh=0.95,restart=False):
        super(OTFForceField,self).__init__()
        self.MLFF=MLFF
        if DFTReqHandler=='VASPSLURM':
            self.DFTReqHandler=FileIOReqHandlerVASP
        elif DFTReqHandler=='Mock':
            from ....Tests.Utils import MockDFTReqHandler
            self.DFTReqHandler=MockDFTReqHandler
        else:
            self.DFTReqHandler=DFTReqHandler
        self.E_thresh=E_thresh
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
    


    def forward(self,atoms,log=True):
        """
        Main on-the-fly force field prediction step.
        
        For each MD step:
        1. Get ML prediction with uncertainty
        2. Calculate confidence metric
        3. If confidence < threshold: request DFT calculation and retrain
        4. Otherwise: use ML prediction
        
        All predictions and reference calculations are saved for analysis.
        
        Args:
            atoms: ASE Atoms object or path to XYZ file
            log: Whether to cache predictions and coordinates
            
        Returns:
            For DFT predictions: [atoms, E_dft, F_dft, S_dft, 0, 0, 0]
            For confident predictions: [atoms, E_ml, F_ml, E_uncert, F_uncert]
        """
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
        if log:
            preds.append(conf)
            torch.save(preds,'ML_preds/{}'.format(self.steps))
            torch.save(atoms,'Coords/{}'.format(self.steps))

        print("Predicted Energy:",E_pred)
        print("Predicted Confidence:", conf)
        print("Confidence Arguments:",self.E_thresh,E_uncert,self.n,self.E)

        if (conf<self.conf_thresh and self.steps>1) or self.steps in [1]:

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
        """
        Update cumulative energy statistics for confidence calculation.
        
        Computes and accumulates squared energy errors relative to initial
        offset prediction, using error-weighted contributions. Used for
        confidence metric updates.
        
        Args:
            new_data: [atoms, E_dft, F_dft] or [atoms, E_dft, F_dft, S_dft]
        """
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


    def update(self,new_data):
        """
        Trigger full retraining cycle with new labeled DFT data.
        
        Coordinates recalibration, ensemble updates, and checkpoint saving.
        The cumulative statistics and step counter are persisted to support
        restart functionality.
        
        Args:
            new_data: [atoms, E_dft, F_dft] or [atoms, E_dft, F_dft, S_dft]
        """
        self.recalibrate(new_data)
        if self.FirstForward:
            new_data[1]+=self.E_offset
        if len(new_data)==4 and not self.StressIncluded:
            new_data=new_data[:3]
        self.MLFF.update(new_data)
        OTFParams=(self.E,self.n,self.E_offset,self.steps)
        torch.save(OTFParams,'tmp/OTFParams')
          



