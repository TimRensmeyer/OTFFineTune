import numpy as np
import torch
import ase
from mace.tools import torch_geometric
from mace import data

class weighted_dataloader():

    def __init__(self,device,atomic_numbers,geometries=[],Y_f=[],Y_e=[],Y_s=[],bs=5,r_max=4.0,pbc=True):
        self.geometries=geometries
        self.Y_f=Y_f
        self.Y_e=Y_e
        self.Y_s=Y_s
        self.size=len(geometries)
        self.bs=bs
        self.r_max=r_max
        self.z_table=data.utils.AtomicNumberTable([int(z) for z in atomic_numbers])
        self.pbc=pbc
        self.device=device


        
    def len(self):
        return self.size
    
    def add(self,sample):
        if len(sample)==3:
            ase_atoms,y_e,y_f=sample
            Stress=False
        else:
            ase_atoms,y_e,y_f,y_s=sample
            y_s=torch.tensor(y_s)
            self.Y_s.append(y_s)


        y_f=torch.tensor(y_f)
        y_e=torch.tensor(y_e)

        self.size+=1

        configs=data.utils.config_from_atoms_list([ase_atoms])
        geometry=data.AtomicData.from_config(configs[0], z_table=self.z_table, cutoff=self.r_max)
        self.geometries.append(geometry)
        self.Y_f.append(y_f)
        self.Y_e.append(y_e)


    
    def sample(self):
        #if the batch size is larger than the size of the data set just return the dataset
        device=self.device
        if self.size<=self.bs:
            b_ind=list(range(np.minimum(self.bs,self.size)))
            batch=[self.geometries[ind] for ind in b_ind]
            dl=torch_geometric.dataloader.DataLoader(
                                            dataset=batch,
                                            batch_size=len(batch),
                                            shuffle=False,
                                            drop_last=False)
            X=next(iter(dl))

            f_t=torch.stack([self.Y_f[i].to(device) for i in b_ind])
            e_t=torch.stack([self.Y_e[i].to(device) for i in b_ind])
            s_t=torch.stack([self.Y_s[i].to(device) for i in b_ind])

            return (X,(e_t,f_t,s_t))


            return (self.geometries,(self.Y_e,self.Y_f,self.Y_s))
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
                    q=L/(ds_size+L)           #p/q=1/L *ds_size/(ds_size+L)
                    weight=c*(p/q)
                    weights.append(weight)
            weights=np.array(weights)
            b_ind=unique
            batch=[self.geometries[ind] for ind in b_ind]
            dl=torch_geometric.dataloader.DataLoader(
                                            dataset=batch,
                                            batch_size=len(batch),
                                            shuffle=False,
                                            drop_last=False)
            X=next(iter(dl))

            f_t=torch.stack([self.Y_f[i].to(device) for i in b_ind])
            e_t=torch.stack([self.Y_e[i].to(device) for i in b_ind])
            s_t=torch.stack([self.Y_s[i].to(device) for i in b_ind])

            return (X,(e_t,f_t,s_t),weights)
                    

    def last_added(self):
        device=self.device

        bs=1
        b_ind=self.ind[(self.size-1):self.size]
        batch=[self.geometries[ind] for ind in b_ind]
        dl=torch_geometric.dataloader.DataLoader(
                                        dataset=batch,
                                        batch_size=len(batch),
                                        shuffle=False,
                                        drop_last=False)
        X=next(iter(dl))

        f_t=[self.Y_f[i].to(device) for i in b_ind]
        e_t=[self.Y_e[i].to(device) for i in b_ind]
        s_t=[self.Y_s[i].to(device) for i in b_ind]

        return (X,(e_t,f_t,s_t))


