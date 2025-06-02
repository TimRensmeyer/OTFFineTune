import torch
import torch.nn as nn
import copy
import numpy as np
import abc
from abc import abstractmethod
import time

# The base class for modeling a parametric propability density that
# all models using this library should subclass from.
# The free parameters of the model need to be registered as Pytorch parameters.
# "evaluate" is a required method prescribing how to evaluate the mean logdensity on a data batch.
# The data batch is expected to be of the form (X_batch,Y_batch) wher X_batch and Y_batch are lists of input/target samples.

class StochasticModel(abc.ABC,nn.Module):

    __metaclass__=abc.ABCMeta

    def __init__(self):
        super(StochasticModel,self).__init__()

    @abc.abstractmethod
    def evaluate(self,data):
        ...

# the base class for Monte Carlo Markov Chain based optimizers that all such optimizer classes should
# inherit from.
# The required "step" method implements a single optimization step
# The required "run" method implements an optimization cycle with nsteps steps.
# cyclic optimizers should implement the basic optimization step in the "step" method
# and the specifics at the cyclus boundaries, such as momentum resampling or deviating stepsizes,
# in the run method for better readability

class MCMCOptimizer(abc.ABC):

    __metaclass__=abc.ABCMeta


    def __init__(self):
        super(MCMCOptimizer,self).__init__()

    @abc.abstractmethod
    def step(self,model):
        ...

    @abc.abstractmethod
    def run(self,nsteps,model):
        ...
        
#This class implements the logarithm of a gaussian mean field prior, rescaled by a scaling factor.
#Both mean and std are expected to be iterables with each sample corresponding to the model layer of the model.Parameters() .
#The shapes of the samples have to broadcastable to the corresponding model layer shape.
class GaussianMeanField(nn.Module):

    def __init__(self,mean,std):
        super(GaussianMeanField,self).__init__()
        self.std=std        #standard deviation of the gaussian density
        self.mean=mean      #mean of the gaussian density
    def change_device(self,device):
        for i in range(len(self.std)):
            self.std[i]=self.std[i].to(device)
            self.mean[i]=self.mean[i].to(device)

    def forward(self,model,scaling):
        log_prior=0.0
        for p,m,s in zip(model.parameters(),self.mean,self.std):
            log_prior+=torch.sum(-0.5*((p-m)/s)**2-0.5*torch.log(2*torch.tensor(3.1415926)*s**2))*scaling

        return log_prior
    
#This class implements the mass term of the AMSGrad Algorithm.
#It stops updating after burnin_steps number of steps have occured
    
class amsmass(nn.Module):

    def __init__(self,beta=0.999,eps=1e-5,burnin_steps=900000):
        super(amsmass,self).__init__()
        self.steps=0
        self.eps=eps
        self.beta=beta
        self.square_mass=None
        self.v=None
        self.beta_pow_steps=1
        self.burnin_steps=burnin_steps

    def change_device(self,device):
        if self.square_mass != None:

            for i in range(len(self.v)):
                self.square_mass[i]=self.square_mass[i].to(device)
                self.v[i]=self.v[i].to(device)

    def forward(self,model):

        if self.square_mass == None:
            self.square_mass=[]
            self.v=[]
            inv_mass=[]
            self.beta_pow_steps*=self.beta
            self.steps+=1
            for q in model.parameters():
                if q.grad != None:
                    v=(1-self.beta)*q.grad**2
                    unbiased_v=v/(1-self.beta_pow_steps)
                    self.square_mass.append(v)
                    inv_mass.append(1/(torch.sqrt(unbiased_v)+self.eps))
                    self.v.append(v)
                else:
                    self.square_mass.append(None)
                    inv_mass.append(None)
                    self.v.append(None)                   

        else:
            self.beta_pow_steps*=self.beta
            inv_mass=[]
            self.steps+=1
            for i,(q,v,m) in enumerate(zip(model.parameters(),self.v,self.square_mass)):
                
                if q.grad != None:
                    if self.steps<=self.burnin_steps:
                        vel=(1-self.beta)*q.grad**2 + self.beta*v
                        new_m=torch.maximum(m,vel)
                        self.v[i]=vel
                        self.square_mass[i]=new_m
                    else:
                        new_m=m

                    unbiased_mass=new_m/(1-self.beta_pow_steps)
                    inv_mass.append(1/(torch.sqrt(unbiased_mass)+self.eps))
                else:
                    inv_mass.append(None)

        return inv_mass
    
# This class implements the Optimization Monte Carlo Algorithm:
# alpha is set by default as 0.1
# eps is set as lr*mass^-1 wehere mass^-1 is a callable that takes a backwarded model as input and can be learned during the initial phase of the optimization.
# lr is the leraning rate (by default 0.001)
# if not specified mass^-1 defaults to the amsgrad denominator

class SGHMC(MCMCOptimizer):
    # Inputs:

    # log_prior: a log_prior class that allows a scaling factor
    # model: a StochasticModel class that expects input data of the form (X,Y) wher X and Y are both lists.
    # dataloader: a dataloader class with with
    #                -a sample method that samples a minibatch (X,Y) as lists
    #                -a len method that returns the size of the dataset
    # mass: See above
    # eps: See above
    def __init__(self,log_prior,model,dataloader,inv_mass = amsmass(),lr=0.001,T=1,burnin_steps=0,preheat=0,debias=True):
        super(SGHMC,self).__init__()
        self.steps=1
        self.log_prior=log_prior
        self.lr=lr
        self.alpha = 0.1
        self.beta_pow_t=1
        self.debias=debias
        self.T=T
        self.Var=[]        
        self.inv_mass=inv_mass
        self.dataloader=dataloader
        self.momentum =[]
        self.burnin_steps=burnin_steps
        self.preheat=preheat
        for p in model.parameters():
            size=p.size()
            value=torch.zeros(size=size,device=p.device)
            self.momentum.append(value)
    def zero_momentum(self):

        with torch.no_grad():       
            for i in range(len(self.momentum)):
                self.momentum[i]*=0

    def change_device(self,device):
        self.inv_mass.change_device(device)
        self.log_prior.change_device(device)
        self.dataloader.device=device
        for i in range(len(self.momentum)):
            self.momentum[i]=self.momentum[i].to(device)

    def est_var(self,model,weighted=False,n_batches=20):
        self.Var=[]
        means=[]

        for p in model.parameters():
            size=p.size()
            value=torch.zeros(size=size,device=p.device)
            means.append(value)
            self.Var.append(value)

        for i in range(n_batches):
            batch=self.dataloader.sample()
            if len(batch)==2:
                X_batch,Y_batch=batch
            else:
                X_batch,Y_batch,weights=batch
            batchsize=len(X_batch)
            dataset_size=self.dataloader.len()
            lp=self.log_prior(model,(1.0/dataset_size))
            ldd=model.evaluate(batch)
            (-lp-ldd).backward()
            with torch.no_grad():
                for j,p in enumerate(model.parameters()):
                    
                    means[j]+=(p.grad/n_batches)
                    
                    p.grad=None
                
        for i in range(n_batches):
            batch=self.dataloader.sample()
            if len(batch)==2:
                X_batch,Y_batch=batch
            else:
                X_batch,Y_batch,weights=batch
            batchsize=len(X_batch)
            dataset_size=self.dataloader.len()
            lp=self.log_prior(model,(1.0/dataset_size))
            ldd=model.evaluate(batch)
            (-lp-ldd).backward()
            with torch.no_grad():
                for j,(p,m) in enumerate(zip(model.parameters(),means)):
                    self.Var[j]+=((p.grad-m)**2/((n_batches-1)*(dataset_size**2)))
                    p.grad=None

    def step(self,model,weighted=False,reduce_var=False):
        batch=self.dataloader.sample()
        if len(batch)==2:
            X_batch,Y_batch=batch
        else:
            X_batch,Y_batch,weights=batch
        batchsize=len(X_batch)
        dataset_size=self.dataloader.len()
        lp=self.log_prior(model,(1.0/dataset_size))
        ldd=model.evaluate(batch)
        (-lp-ldd).backward()
        with torch.no_grad():
            inv_mass=self.inv_mass(model)
            self.beta_pow_t*=(1-self.alpha)
            self.steps+=1
            for q,im,p in zip(model.parameters(),inv_mass,self.momentum):
                if q.grad!=None:
                    #Here we allow for an initial phase with no noise added to the process
                    if self.steps< self.burnin_steps:
                        new_p=p-self.alpha*q.grad - self.alpha*p
                        
                    #Here we linearily increase the noise to the desired temperature over the course of a certain amount of preheat steps  
                    elif self.steps < self.burnin_steps+self.preheat:
                        temp=self.T*(self.steps-self.burnin_steps)/self.preheat
                        means=torch.zeros(size=q.size(),device=q.device)
                        std=torch.sqrt(2*(1/self.lr)*(1/im)*(1/dataset_size))*temp

                        noise=torch.normal(mean=means,std=std)
                        new_p=p-self.alpha*q.grad - self.alpha*p + self.alpha*noise
                    
                    #Running the algorithm step at the specified temperature. 
                    #At self.T=1 this will converge to the Bayesian posterior for appropriate learning rates.
                    else:
                        temp=self.T
                        means=torch.zeros(size=q.size(),device=q.device)
                        std=torch.sqrt(2*(1/self.lr)*(1/im)*(1/dataset_size))*temp

                            
                        noise=torch.normal(mean=means,std=std)
                        new_p=p-self.alpha*q.grad - self.alpha*p + self.alpha*noise
                                 
                    p.copy_(new_p)
                    p.grad=None
                    
                    #Optionally debiasing the momentum term like in the Adam Algorithm
                    if self.debias:
                        denom=(1-self.beta_pow_t)
                    else:
                        denom=1
                    
                    new_q=q+self.lr*im*new_p/denom
                    q.copy_(new_q)
                    q.grad=None


    def run(self,nsteps,model,avg_model=None):
        for i in range(nsteps):
            self.step(model)
            
            #Optional for keeping a model with the exponential moving average of the weights
            if avg_model != None:
                for (q,p) in zip(model.parameters(),avg_model.parameters()):
                    if q.grad != None:
                        print('gradients not zeroed')
                    p.requires_grad=False
                    new_p=(1-self.lr)*p+self.lr*q.detach()
                    new_p.requires_grad=False
                    p.copy_(new_p)
                    p.requires_grad=True       
        
        return avg_model

class CyclicOptimizer():

    def __init__(self, model,log_prior,dataloader,cycle_length=20, max_lr=0.003):
        model_device=next(iter(model.parameters())).device
        
        self.optimizer=SGHMC(log_prior,model,dataloader,burnin_steps=400,preheat=100,debias=False)
        self.cycle_length=cycle_length
        self.max_learning_rate=max_lr
        self.initialized=False

    def change_device(self,device):
        self.optimizer.change_device(device)

    def add(self,new_data):
        self.optimizer.dataloader.add(new_data)

    def run(self,model):
        alpha=self.optimizer.alpha
        ds_size=self.optimizer.dataloader.len() 
        self.optimizer.inv_mass=amsmass()
        eps=1e-4
        self.optimizer.inv_mass.eps=eps
        self.optimizer.steps=1
        bs=self.optimizer.dataloader.bs

        if not self.initialized: #Option if the first Markov Chain is supposed to be longer to generate better initial seeds. In practice we found no improvement so far.
            self.initialized=True
            for i in range(0,self.cycle_length):

                lr=0.5*self.max_learning_rate*(1+np.cos(i*3.1415926/(self.cycle_length)))+1e-8
                self.optimizer.lr=lr
                self.optimizer.step(model)#=mini_batching)
        else:
            for i in range(0,self.cycle_length):
                lr=0.5*self.max_learning_rate*(1+np.cos(i*3.1415926/(self.cycle_length)))+1e-8
                self.optimizer.lr=lr
                self.optimizer.step(model)

        self.optimizer.zero_momentum()

        return model