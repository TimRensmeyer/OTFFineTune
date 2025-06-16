import torch
from torch import nn as nn
from mace.calculators import mace_mp, mace_off
from mace.tools import torch_geometric
from mace import data
import yaml


from DataLoader import weighted_dataloader
from MCMC import CyclicOptimizer, GaussianMeanField

def init_weights_zeros(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        model = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cpu',return_raw_model=True)
        model.float()
        self.model=model

        self.z_table=data.utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self.E_uncert=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU(),nn.Linear(16,1))
        self.F_uncert=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU(),nn.Linear(16,1))
        self.E_uncert.apply(init_weights_zeros)
        self.F_uncert.apply(init_weights_zeros)
        self.S_uncert=nn.Parameter(torch.zeros(1,),requires_grad=True)

    def forward(self, atoms_list,training=False):
        #In a normal forward call the Input will be a list of ASE Atoms objects
        if isinstance(atoms_list, list):
            configs=data.utils.config_from_atoms_list(atoms_list)
            batch=[data.AtomicData.from_config(cfg, z_table=self.z_table, cutoff=self.model.r_max.item())
                for cfg in configs]
            dl=torch_geometric.dataloader.DataLoader(
                                                        dataset=batch,
                                                        batch_size=len(batch),
                                                        shuffle=False,
                                                        drop_last=False)
            X=next(iter(dl))
            
        # During Training the dataloader will store the input data as a list of AtomicData objects, so the geometric graph won't have to be create each time a geometry is sampled
        # Hence the samples will already be in the right structure.
        else:
            X=atoms_list


        device=next(iter(self.model.parameters())).device
        X=X.to(device)
        out=self.model(X.to_dict(),compute_stress = True,training=training)
        stress=out['stress']
        forces=out['forces']
        energy=out['energy']
        device=energy.device
        node_feats=torch.cat((out['node_feats'][:,:128],out['node_feats'][:,-128:]),dim=1)
        e_stds=torch.exp(self.E_uncert(node_feats)).squeeze(-1)*0+0.6
        
        force_uncert=torch.exp(self.F_uncert(node_feats)).squeeze(-1)*0.1
        force_uncert=torch.stack([force_uncert]*3,dim=-1)
        stress_uncert=torch.ones(size=stress.size(), device=device)*torch.exp(self.S_uncert)*0+0.1/16

        indices=[]
        for i in range(X.ptr.shape[0]-1):
            indices+=[i]*(X.ptr[i+1]-X.ptr[i])
        
        n_samples=X.ptr.shape[0]-1
        E_mol=torch.zeros((n_samples,)).to(device)
        energy_uncert=torch.zeros((n_samples,)).to(device)
        n_atoms=torch.zeros((n_samples,)).to(device)
        energy_uncert.scatter_add_(src=e_stds,index=torch.tensor(indices).long().to(device),dim=0)
        n_atoms.scatter_add_(src=e_stds*0+1,index=torch.tensor(indices).long().to(device),dim=0)
        energy_uncert/=n_atoms

        return energy,forces,stress,energy_uncert,force_uncert,stress_uncert


class smodel(nn.Module):
    def __init__(self):
        super(smodel,self).__init__()
        self.net=Network()

    def predict(self,atoms_list,training=False):
        energy,forces,stress,energy_uncert,force_uncert,stress_uncert=self.net(atoms_list,training=training)
        return energy*23.0609,forces*23.0609,stress*23.0609,(energy_uncert,force_uncert,stress_uncert)
    
    def evaluate(self,data):
        dev=next(iter(self.net.parameters())).device
        atoms_list=data[0]
        
        
        (dft_energy,dft_force,dft_stress)=data[1]
        batch_size=dft_energy.shape[0]
        energy,forces,stress,(energy_uncert,force_uncert,stress_uncert)=self.predict(atoms_list,training=True)

        weighted=False
        if len(data)==3:
            weighted=True
            weights=torch.tensor(data[2]).to(dev)


        if weighted:
            force_weights=[]
            for f,w in zip(dft_force,weights):
                force_weights.append((f*0+w).detach())

            force_weights=[fw.reshape(-1,3) for fw in force_weights]
            force_weights=torch.cat(force_weights).to(dev)  

            stress_weights=[]
            for s,w in zip(dft_stress,weights):
                stress_weights.append(torch.zeros(size=(3,3)).to(dev)+w)
            stress_weights=torch.stack(stress_weights).to(dev) 

        dft_force=torch.cat([f for f in dft_force])
        dft_stress=torch.stack([s for s in dft_stress])
        if batch_size>1:
            dft_energy=torch.cat([e for e in dft_energy])
        else:
            dft_energy=dft_energy[0]
        
        exponent_e=-0.5*(energy-dft_energy)**2/energy_uncert**2
        exponent_f=-0.5*(forces-dft_force)**2/force_uncert**2
        exponent_s=-0.5*(stress-dft_stress)**2/stress_uncert**2
        print(torch.mean(torch.abs((energy-dft_energy))).detach().cpu().item(),
              torch.mean(torch.abs((forces-dft_force))).detach().cpu().item(),
              torch.mean(torch.abs((stress-dft_stress))).detach().cpu().item(),
              torch.mean(torch.abs((energy_uncert))).detach().cpu().item())

        ll_e=exponent_e-0.5*torch.log(2*3.1415926*energy_uncert**2)
        ll_f=exponent_f-0.5*torch.log(2*3.1415926*force_uncert**2)
        ll_s=exponent_s-0.5*torch.log(2*3.1415926*stress_uncert**2)
        
        if not weighted:

            ll_e=torch.sum(ll_e)/batch_size
            ll_f=torch.sum(ll_f)/batch_size
            ll_s=torch.sum(ll_s)/batch_size


        else:
            ll_e=torch.sum(ll_e*weights)/batch_size
            ll_f=torch.sum(ll_f*force_weights)/batch_size
            ll_s=torch.sum(ll_s*stress_weights)/batch_size     
                   

        return ll_e+ll_f+ll_s



class MACE_Wrapper(nn.Module):
    def __init__(self,args):
        super(MACE_Wrapper,self).__init__()
        prior_strength=args[0]
        self.model=smodel()
        mean=[]
        std=[]
        i=0
        for p in self.model.parameters():
            mean.append(p.detach())
            std.append(p.detach()*0+1*prior_strength)
        self.log_prior=GaussianMeanField(mean,std)
        atomic_numbers=self.model.net.model.atomic_numbers
        cutoff=self.model.net.model.r_max

        dataloader=weighted_dataloader(atomic_numbers=atomic_numbers,
                                       bs=5,
                                       device=torch.device("cpu"),
                                       r_max=cutoff.cpu().item())
        
        self.optimizer=CyclicOptimizer(self.model,self.log_prior,
                                       dataloader=dataloader,
                                       cycle_length=2000,
                                         max_lr=0.001)
        
    def predict(self,ase_atoms):
        out=self.model.predict([ase_atoms])
        return [out[0],out[1],out[2],out[3][0],out[3][1],out[3][2]]
    
    def change_device(self,device):
        self.optimizer.change_device(device)
        self.model=self.model.to(device)
    
    def update(self,new_data):
        self.optimizer.add(new_data)
        self.model=self.optimizer.run(self.model)

def MACE_Builder(args):
    return MACE_Wrapper(args)

