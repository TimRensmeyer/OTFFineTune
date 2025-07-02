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
import shutup
shutup.please()
with open('runconfig.yaml', 'r') as file:
    config = yaml.safe_load(file)

CodePath=config['CodePath']
ErrorThreshold=config['ErrorThreshold']
ErrorThreshold
sys.path.insert(0, CodePath)

from OTFFineTune.MCMC import GaussianMeanField,CyclicOptimizer


conf=Config()
conf=conf.from_file(CodePath+'OTFFineTune/config.yaml')
model=model_from_config(conf,initialize=True).model
base=nn.Sequential(*[model.func[i] for i in range(len(model.func)-4)])

class Network(nn.Module):
    def __init__(self,dict_size=4):
        super(Network,self).__init__()
        self.rescale=nn.Parameter(torch.ones(dict_size,),requires_grad=True)
        self.lin=nn.Sequential(nn.Linear(64,32),nn.Linear(32,1))
        self.linf=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        self.line=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
        self.func=copy.deepcopy(base)


        
    def forward(self,X):
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

from MCMC import StochasticModel
class model(StochasticModel):
    def __init__(self,net,scale=14.3117/0.529177):
        super(model,self).__init__()
        self.net=net
        self.scale=scale
    
    def predict(self,Atoms,R,Lattice=None):
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


    def evaluate(self,data):
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


def NequIP_Loader():

    module=Network(dict_size=100)
    SpiceDict=torch.load(CodePath +'OTFFineTune/Dicts/SpiceDict',map_location=torch.device('cpu'))
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

from OTFFineTune.NNP import NNP
from OTFFineTune.NequIPDataLoader import weighted_dataloader
class NequIP_Wrapper(NNP):
    def __init__(self,args):
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
        dataloader=weighted_dataloader(bs=5,device=torch.device("cpu"))
        self.optimizer=CyclicOptimizer(self.model,self.log_prior,
                                       dataloader=dataloader, max_lr=0.0001,cycle_length=2000)

    def predict(self,ase_atoms):
        R=ase_atoms.get_positions()
        Atoms=ase_atoms.get_atomic_numbers()
        e_pred,f_pred,(std_e,std_f)=self.model.predict(Atoms,R)

        return (e_pred,f_pred,std_e,std_f)
    
    def change_device(self,device):
        self.optimizer.change_device(device)
        self.model=self.model.to(device)

    def update(self,new_data):
        self.optimizer.add(new_data)
        self.model=self.optimizer.run(self.model)

def NequIP_Builder(args):
    return NequIP_Wrapper(args)