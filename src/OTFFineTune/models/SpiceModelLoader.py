"""
SPICE/NequIP Potential Wrapper with Uncertainty Quantification

This module wraps a NequIP model that was pretrained on the
SPICE Dataset (foundation model) for use in the OTF fine-tuning framework.


The Network class extends the pre-trained model with trainable uncertainty heads.
The wrapper handles model predictions, likelihood computation, 
and training for all supported elements.

Key Features:
- Aleatoric uncertainty heads for energy and forces
- Integration with SGHMC sampling via CyclicOptimizer
- Trainable uncertainty estimates via neural network heads
- Gaussian negative log-likelihood loss for Bayesian sampling
"""
import shutup
shutup.please()
import os
import nequip
from nequip.utils.config import Config
import torch
import torch.nn as nn
from nequip.utils.config import Config
from nequip.data.AtomicData import AtomicData
from nequip.data import AtomicDataDict
from nequip.model._build import model_from_config
from nequip.utils.torch_geometric import Batch
import copy
import yaml
import sys
import numpy as np
from typing import List, Union, Any, Optional

with open('runconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)


ErrorThreshold=config['ErrorThreshold']


from ..core.MCMC import GaussianMeanField,CyclicOptimizer

file_path=os.path.dirname(os.path.realpath(__file__))
config_path=os.path.join(file_path,'constructor_data/config.yaml')
model_dict_path=os.path.join(file_path,'constructor_data/SpiceDict')
conf=Config()
conf=conf.from_file(config_path)

model=model_from_config(conf,initialize=True).model

base=nn.Sequential(*[model.func[i] for i in range(len(model.func)-4)])

class Network(nn.Module):
    """
    NequIP-based neural network with uncertainty quantification.
    
    Extends pretrained SPICE foundation model with:
    - Trainable energy uncertainty head
    - Trainable force uncertainty head
    
    Uses the Batch class from the NequIP library for input graph representation.
    Uncertainty heads are initialized to output small uncertainties.
    
    Args:
        dict_size: Number of element types (default 100)
    """
    def __init__(self,dict_size: int = 4) -> None:
        super(Network,self).__init__()
        self.rescale=nn.Parameter(torch.ones(dict_size,),requires_grad=True)
        self.lin=nn.Sequential(nn.Linear(64,32),nn.Linear(32,1))
        self.linf=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        self.line=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        self.func=copy.deepcopy(base)


        
    def forward(self,X: List) -> List:
        (type_batch,diction)=X
        dev=next(iter(self.lin.parameters())).device

        diction['atom_types']=type_batch.to(dev)
        diction['pos']=diction['pos'].detach().to(dev)
        diction['edge_index']=diction['edge_index'].to(dev)
        diction['cell']=diction['cell'].to(dev)
        diction['edge_cell_shift']=diction['edge_cell_shift'].to(dev)
        diction['pos'].requires_grad=True

        atomic_data=self.func(diction)
        ptr=atomic_data['ptr']
        z=atomic_data['node_features'][:,0:64]    
        E=self.lin(z)
        std_e=self.line(z)
        std_f=self.linf(z)
        std_e=torch.exp(std_e)*0+ErrorThreshold*0.1
        std_f=torch.exp(torch.stack([std_f]*3,dim=1))*0.1

        rescale=self.rescale[type_batch.long()].unsqueeze(1) 
                                                             
        E=E*rescale   
        indices=[]
        for i in range(ptr.shape[0]-1):
            indices+=[i]*(ptr[i+1]-ptr[i])
        
        
        n_samples=ptr.shape[0]-1
        E_mol=torch.zeros((n_samples,)).to(dev)
        std_e_mol=torch.zeros((n_samples,)).to(dev)
        n_atoms=torch.zeros((n_samples,)).to(dev)
        E_mol.scatter_add_(src=E.squeeze(-1),index=torch.tensor(indices).long().to(dev),dim=0)
        std_e_mol.scatter_add_(src=std_e.squeeze(-1),index=torch.tensor(indices).long().to(dev),dim=0)
        n_atoms.scatter_add_(src=std_e.squeeze(-1)*0+1,index=torch.tensor(indices).long().to(dev),dim=0)
        std_e_mol/=n_atoms
        
        E_list=[E_mol[j] for j in range(ptr.shape[0]-1)]
        F=torch.autograd.grad(E_list,diction['pos'],retain_graph=True,create_graph=True)
        return E_mol.unsqueeze(1),-torch.cat(F),[std_e_mol.unsqueeze(1),std_f.squeeze(-1)]

from ..core.MCMC import StochasticModel
class model(StochasticModel):
    """
    Probabilistic model wrapper for the Network class above.
    
    Provides training-ready interface with:
    - predict(): Generate predictions with uncertainties
    - evaluate(): Compute Gaussian negative log-likelihood
    
    Converts ASE atomic structures to NequIP graph representation for inference.
    Handles unit conversions between the SPICE dataset (atomic units) and internal units.
    """
    def __init__(self,
                 net: nn.Module,
                 scale: float = 14.3117/0.529177) -> None:
        super(model,self).__init__()
        self.net=net
        self.scale=scale
    
    def predict(self,
                Atoms: List[int],
                R=Union[np.array, torch.Tensor],
                Lattice=Optional[Union[np.array,torch.Tensor]]) -> List:
        """
        Generate single prediction with uncertainties.
        
        Args:
            Atoms: Array of atomic numbers
            R: Position array in Ångströms
            Lattice: (unused) for API compatibility
            
        Returns:
            (energy, forces, (energy_std, force_std)) in kcal/mol and kcal/molÅ
        """
        dev=next(iter(self.net.parameters())).device
        R=torch.tensor(R).to(dev)
        x_a=torch.tensor(Atoms).long().to(dev)
        R=AtomicData.from_points(R,r_max=4,pbc=False).to(dev)
        R_batched=Batch.from_data_list([R]).to(dev)
        R=R.to_AtomicDataDict(R_batched)
        e_pred,f_pred,(std_e,std_f)=self.net((x_a,R))
        scale=self.scale
        e_pred*=scale
        f_pred*=scale

        
        return e_pred,f_pred,(std_e,std_f)


    def evaluate(self,data: List) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood on mini-batch.
        
        Supports weighted importance sampling for unbiasing gradient estimates 
        when using a wweighted dataloader.
        
        Args:
            data: (X_batch, (Y_e, Y_f)) or (X_batch, (Y_e, Y_f), weights)
                  - X_batch: Batch class from the NequIP library
                  - Y_e: Energy targets (batch_size,)
                  - Y_f: Force targets, list of (n_atoms_i, 3)
                  - weights: Importance sampling weights (optional)
        """
        dev=next(iter(self.net.parameters())).device
        scale=self.scale
        X=data[0]
        Y=data[1]
        weighted=False
        if len(data)==3:
            weighted=True
            weights=torch.tensor(data[2]).to(dev)

        Y_e=torch.stack(Y[0]).to(dev).squeeze(-1)
        Y_f=Y[1]
        if weighted:
            force_weights=[]
            for f,w in zip(Y_f,weights):
                force_weights.append((f*0+w).detach())

            force_weights=[fw.reshape(-1,3) for fw in force_weights]
            force_weights=torch.cat(force_weights).to(dev)  

        Y_f=[f.reshape(-1,3) for f in Y_f]
        Y_f=torch.cat(Y_f).to(dev)  
        pred_e,pred_f,stds=self.net(X)  
        bs=len(X[1]['ptr'])-1
        std_e=stds[0]
        std_f=stds[1]
        exponent_e=-0.5*((Y_e.squeeze(1)-scale*pred_e.squeeze(1))/(std_e.squeeze(1)))**2
        exponent_f=-0.5*((Y_f-scale*pred_f.squeeze(0))/std_f)**2
        e_error=torch.mean(torch.abs((Y_e.squeeze(1)-scale*pred_e.squeeze(1))))
        f_error=torch.mean(torch.abs((Y_f-scale*pred_f.squeeze(0))))
        print('energy error:',e_error.detach().cpu().item(),'force error:',f_error.detach().cpu().item(),
              "energy std:",torch.mean(std_e).detach().cpu().item())
        if not weighted:
            print(std_f.shape,Y_f.shape,pred_f.squeeze(0).shape)
        else:
            print(std_f.shape,Y_f.shape,pred_f.squeeze(0).shape,force_weights.shape,weights.shape)

        ll_e=exponent_e-0.5*torch.log(2*3.1415926*std_e.squeeze(1)**2)
        ll_f=exponent_f-0.5*torch.log(2*3.1415926*std_f**2)

        if weighted:
            return torch.sum(ll_f*force_weights)/bs+torch.sum(ll_e*weights)/bs
        else:
            return torch.sum(ll_f)/bs+torch.sum(ll_e)/bs   


def NequIP_Loader() -> StochasticModel:
    """
    Initialize NequIP model with pretrained weights from checkpoint.
    
    Loads foundation model weights from SpiceDict checkpoint, preserving
    pretrained features while zeroing out uncertainty heads for transfer learning.
    
    Returns:
        model: Initialized probabilistic model with uncertainty quantification
    """
    module=Network(dict_size=100)
    SpiceDict=torch.load(model_dict_path,map_location=torch.device('cpu'))
    keys=SpiceDict.keys()
    dict={}
    for k in keys:
        if ('linf.' in k) or ('line.' in k):
            new_k='net'+k[6:]
            dict[new_k]=SpiceDict[k]*0
        else:
            new_k='net'+k[6:]
            dict[new_k]=SpiceDict[k]
    m=model(module, scale=23.06)
    m.load_state_dict(state_dict=dict)    

    return copy.deepcopy(m)

from ..core.NNP import NNP
from ..data.NequIPDataLoader import weighted_dataloader
import ase
class NequIP_Wrapper(NNP):
    """
    Complete training wrapper for SPICE potential with on-the-fly fine-tuning
    for integration into the NNP.py structure.
    
    Integrates Network, model, and CyclicOptimizer for Bayesian transfer learning.
    Sets up selective Gaussian priors:
    - Weak priors on uncertainty head parameters (layers 5-16)
    - Strong priors on foundation model parameters for stability
    
    Args:
        args: [prior_strength] - strength of Gaussian prior on parameters
    """
    def __init__(self,
                 args: List,
                 testing: bool = False) -> None:
        super(NequIP_Wrapper,self).__init__()
        prior_strength=args[0]
        self.model=NequIP_Loader()
        mean=[]
        std=[]
        i=0
        for p in self.model.parameters():

            if i>=5 and i<=16:
                mean.append(p.detach()*0)
                std.append(p.detach()*0+1*prior_strength)
            else:
                mean.append(p.detach())
                std.append(p.detach()*0+prior_strength)
            i+=1

        self.log_prior=GaussianMeanField(mean,std)
        if testing:
            bs=2
        else:
            bs=5
        dataloader=weighted_dataloader(bs=bs,device=torch.device("cpu"))
        if not testing:
            self.optimizer=CyclicOptimizer(self.model,self.log_prior,
                                        dataloader=dataloader, max_lr=0.0001,cycle_length=2000)
        else:
            self.optimizer=CyclicOptimizer(self.model,self.log_prior,
                                        dataloader=dataloader, max_lr=0.0001,cycle_length=2,
                                        burnin_steps=0,preheat=0)

    def predict(self,ase_atoms: ase.Atoms) -> List:
        """Generate single prediction with uncertainties from ASE Atoms object."""
        R=ase_atoms.get_positions()
        Atoms=ase_atoms.get_atomic_numbers()
        e_pred,f_pred,(std_e,std_f)=self.model.predict(Atoms,R)

        return (e_pred,f_pred,std_e,std_f)
    
    def change_device(self,device: torch.device) -> None:
        """Move model and optimizer to device."""
        self.optimizer.change_device(device)
        self.model=self.model.to(device)

    def update(self,new_data: List) -> None:
        """Retrain with new labeled data using CyclicOptimizer."""
        self.optimizer.add(new_data)
        self.model=self.optimizer.run(self.model)

def NequIP_Builder(args,testing=False) -> NequIP_Wrapper:
    """Factory function to create NequIP_Wrapper instance."""
    return NequIP_Wrapper(args,testing=testing)