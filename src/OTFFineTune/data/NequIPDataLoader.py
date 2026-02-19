"""
Data Loaders for NequIP Models

This module provides data loading utilities specifically for NequIP model training.
The weighted_dataloader implements importance sampling to emphasize newly added
on-the-fly samples.

Note: The simple dataloader class is currently no longer used and may be removed 
in future versions. Use weighted_dataloader for new implementations.

Classes:
    dataloader: Basic batch sampler (deprecated)
    weighted_dataloader: Batch sampler with importance weighting
"""

import numpy as np
import torch
import ase
from nequip.data.AtomicData import AtomicData
from nequip.data import AtomicDataDict
from nequip.model._build import model_from_config
from nequip.utils.torch_geometric import Batch
from typing import List, Any, Union, Optional

class dataloader():
    """
    Basic Data Loader for NequIP models.
    
    Note: This class is currently no longer used and may be removed in future versions.
    Use weighted_dataloader for new code.
    
    Converts ASE Atoms to NequIP graph representation with simple random sampling.
    """

    def __init__(self,
                 device: torch.device,
                 X_pos: List = [],
                 X_a: List = [],
                 Y_f: List = [],
                 Y_e: List = [],
                 bs: int = 5,
                 r_max: float = 4,
                 pbc: bool = False) -> None:
        self.X_pos=X_pos
        self.X_a=X_a
        self.Y_f=Y_f
        self.Y_e=Y_e
        self.size=len(X_a)
        self.counter=1
        self.bs=bs
        self.r_max=r_max
        self.pbc=pbc
        self.ind = np.arange(self.size)
        self.device=device
        np.random.shuffle(self.ind)

        
    def len(self) -> int:
        return self.size
    
    def add(self, sample: List) -> None:

        ase_atoms,y_e,y_f=sample
        R=ase_atoms.get_positions()
        x_a=ase_atoms.get_atomic_numbers()
        R=torch.tensor(R)
        y_f=torch.tensor(y_f)
        y_e=torch.tensor(y_e)
        x_a=torch.tensor(x_a)
        R=AtomicData.from_points(R,r_max=4,pbc=False)

        self.size+=1
        self.ind = np.arange(self.size)
        np.random.shuffle(self.ind)
        self.X_pos.append(R)
        self.X_a.append(x_a)
        self.Y_f.append(y_f)
        self.Y_e.append(y_e)

    
    def sample(self) -> List:
        device=self.device
        if self.bs*self.counter > self.size:
            np.random.shuffle(self.ind)
            self.counter=1
        bs=np.minimum(self.bs,self.size)
        b_ind=self.ind[(self.counter-1)*bs:self.counter*bs]
        R=Batch.from_data_list([self.X_pos[ind].to(device) for ind in b_ind])
        R=self.X_pos[0].to_AtomicDataDict(R)

        a=torch.cat([self.X_a[i].to(device) for i in b_ind])

        f_t=[self.Y_f[i].to(device) for i in b_ind]
        e_t=[self.Y_e[i].to(device) for i in b_ind]
        self.counter+=1

        return ((a,R),(e_t,f_t))
    
    def last_added(self) -> List:
        device=self.device

        bs=1
        b_ind=self.ind[(self.size-1):self.size]
        R=Batch.from_data_list([self.X_pos[ind].to(device) for ind in b_ind])
        R=self.X_pos[0].to_AtomicDataDict(R)

        a=torch.cat([self.X_a[i].to(device) for i in b_ind]).long()
        f_t=torch.cat([self.Y_f[i].to(device) for i in b_ind])
        e_t=torch.stack([self.Y_e[i].to(device) for i in b_ind])


        return ((a,R),(e_t,f_t))
    
class weighted_dataloader():
    """
    Weighted Data Loader for importance-sampled mini-batch training.
    
    Implements importance sampling to emphasize newly added training samples
    while maintaining statistical validity. Converts ASE structures to NequIP
    graph representation and supports weighted sampling.
    
    Args:
        device: PyTorch device
        X_pos: List of position graphs (AtomicData)
        X_a: List of atomic number lists
        Y_f: List of force targets
        Y_e: List of energy targets
        bs: Batch size
        r_max: Interaction cutoff radius
        pbc: Whether periodic boundary conditions apply
    """

    def __init__(self,
                 device: torch.device,
                 X_pos: List = [],
                 X_a: List = [],
                 Y_f: List = [],
                 Y_e: List = [],
                 bs: int = 5,
                 r_max: float = 4,
                 pbc: bool = False) -> None:
        
        self.X_pos=X_pos
        self.X_a=X_a
        self.Y_f=Y_f
        self.Y_e=Y_e
        self.size=len(X_a)
        self.counter=1
        self.bs=bs
        self.r_max=r_max
        self.pbc=pbc
        self.ind = np.arange(self.size)
        self.device=device
        np.random.shuffle(self.ind)

        
    def len(self) -> int:
        return self.size
    
    def add(self,sample: List) -> None:

        ase_atoms,y_e,y_f=sample
        R=ase_atoms.get_positions()
        x_a=ase_atoms.get_atomic_numbers()
        R=torch.tensor(R)
        y_f=torch.tensor(y_f)
        y_e=torch.tensor(y_e)
        x_a=torch.tensor(x_a)
        R=AtomicData.from_points(R,r_max=4,pbc=False)

        self.size+=1
        self.ind = np.arange(self.size)
        np.random.shuffle(self.ind)
        self.X_pos.append(R)
        self.X_a.append(x_a)
        self.Y_f.append(y_f)
        self.Y_e.append(y_e)

    
    def sample(self) -> List:
        #if the batch size is larger than the size of the data set just return the dataset
        device=self.device
        if self.size<=self.bs:
            
            if self.bs*self.counter > self.size:
                np.random.shuffle(self.ind)
                self.counter=1
            bs=np.minimum(self.bs,self.size)
            b_ind=self.ind[(self.counter-1)*bs:self.counter*bs]
            R=Batch.from_data_list([self.X_pos[ind].to(device) for ind in b_ind])
            R=self.X_pos[0].to_AtomicDataDict(R)

            a=torch.cat([self.X_a[i].to(device) for i in b_ind])

            f_t=[self.Y_f[i].to(device) for i in b_ind]
            e_t=[self.Y_e[i].to(device) for i in b_ind]
            self.counter+=1

            return ((a,R),(e_t,f_t))
        else:
            #If the batch size is smaller than the size of the dataset perform importance sampling
            # to prioritize the newly added sample.
            # for a batch size bs, a dataset size ds and a sampling probability sp for the 
            # the chance of sampling the new sample in the batch is 1-(1-sp)^bs.
            # on average there will be sp*bs replicas of the new sample in the batch.
            # to avoid redundance it makes sense to set sp=1/bs. The the chance of not sampling 
            # the new sample asymptotically aproaches 1-1/e\approx 0.63 from above for growing batch sizes
            # the sampling probability can approximately be achieved by sampling from a set
            # S that contains each sample 1 time except the last which is contained L times.
            # then sp = L/(S+L)~1/bs => L~=(S+L)/bs => L(1-1/bs)~=S/bs =>L~=S/(bs(1-1/bs))=S/(bs-1)
            ds_size=self.size   
            bs=self.bs
            L=np.ceil(ds_size/(bs-1))
            samples=np.random.randint(low=0, high=ds_size+L,size=bs) #sampling batch 
            clamped_samples=[]
            for s in samples:
                if s<ds_size-1:
                    clamped_samples.append(s)
                else:
                    clamped_samples.append(ds_size-1)
            samples=np.array(clamped_samples)
            #reducing redundant samples by weighting
            unique, counts = np.unique(samples, return_counts=True) 
            weights=[]
            for s,c in zip(unique,counts):
                if s<ds_size-1:
                    p=1/ds_size
                    q=1/(ds_size+L)
                    weight=c*(p/q)
                    weights.append(weight)
                else:
                    p=1/ds_size
                    q=L/(ds_size+L)
                    weight=c*(p/q)
                    weights.append(weight)
            weights=np.array(weights)
            b_ind=unique
            R=Batch.from_data_list([self.X_pos[ind].to(device) for ind in b_ind])
            R=self.X_pos[0].to_AtomicDataDict(R)

            a=torch.cat([self.X_a[i].to(device) for i in b_ind])

            f_t=[self.Y_f[i].to(device) for i in b_ind]
            e_t=[self.Y_e[i].to(device) for i in b_ind]
            self.counter+=1

            return ((a,R),(e_t,f_t),weights)
                    

    def last_added(self) -> List:
        device=self.device

        bs=1
        b_ind=self.ind[(self.size-1):self.size]
        R=Batch.from_data_list([self.X_pos[ind].to(device) for ind in b_ind])
        R=self.X_pos[0].to_AtomicDataDict(R)

        a=torch.cat([self.X_a[i].to(device) for i in b_ind]).long()
        f_t=torch.cat([self.Y_f[i].to(device) for i in b_ind])
        e_t=torch.stack([self.Y_e[i].to(device) for i in b_ind])


        return ((a,R),(e_t,f_t))