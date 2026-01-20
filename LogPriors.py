"""
Log-Prior Distributions for Bayesian Neural Networks

All log-priors are callable objects that compute log p(θ) given a model.
They enable:
- Transfer learning: Gaussian priors centered on pretrained weights
- Regularization: Constraining model parameters to prior expectations
- Hyperpriors: Hierarchical priors on hyperparameters

Log-priors accept a scaling factor to adjust their strength in the posterior
(useful for importance weighting in mini-batch training).

Classes:
    GaussianMeanField: Gaussian prior with per-layer means and variances
    LaplaceMeanField: Laplace (L1) prior
    TransferLearningPrior: Gaussian prior relative to another model
    HyperPrior: Empirical Bayes prior combining model and data likelihood
    TransferLearningHyperPrior: Hierarchical prior for transfer learning
"""

# All logpriors are callable objects, that take a nn.Module subclass as input and return the (rescaled) logarithm of the prior of the model
import torch
import torch.nn as nn

#This class implements the logarithm of a gaussian mean field prior, rescaled by a scaling factor.
#Both mean and std are expected to be iterables with each sample corresponding to the model layer of the model.Parameters() .
#The shapes of the samples have to broadcastable to the corresponding model layer shape.
class GaussianMeanField(nn.Module):
    """
    Gaussian mean field log-prior for transfer learning.
    
    Implements log p(θ) = -0.5 Σ((θ_i - μ_i) / σ_i)² - 0.5 Σ log(2π σ_i²)
    
    Args:
        mean: List of prior means for each parameter (typically pretrained weights)
        std: List of prior standard deviations for each parameter
    """

    def __init__(self,mean,std):
        super(GaussianMeanField,self).__init__()
        self.std=std        #standard deviation of the gaussian density
        self.mean=mean      #mean of the gaussian density

    def forward(self,model,scaling):
        log_prior=0.0
        for p,m,s in zip(model.parameters(),self.mean,self.std):
            dev=p.device
            m=m.to(dev).detach()
            s=s.to(dev).detach()
            log_prior+=torch.sum(-0.5*((p-m)/s)**2-0.5*torch.log(2*torch.tensor(3.1415926).to(dev)*s**2))*scaling

        return log_prior
    
    
class LaplaceMeanField(nn.Module):

    def __init__(self,mean,std):
        super(LaplaceMeanField,self).__init__()
        self.std=std        #standard deviation of the gaussian density
        self.mean=mean      #mean of the gaussian density

    def forward(self,model,scaling):
        """
        Laplace (L1) prior: log p(θ) = -Σ|θ_i - μ_i|/σ_i - Σ log(2σ_i)
        """
        log_prior=0.0
        for p,m,s in zip(model.parameters(),self.mean,self.std):
            dev=p.device
            m.to(dev)
            s.to(dev)
            log_prior+=torch.sum(-torch.abs(p-m)/s-torch.log(2*torch.tensor(s)))*scaling

        return log_prior
    
class TransferLearningPrior(nn.Module):

    def __init__(self,std):
        super(TransferLearningPrior,self).__init__()
        self.std=std           
        
    def forward(self,model,p_model,scaling):
        """
        Transfer learning prior: Gaussian centered on another model's weights.
        
        Used in hierarchical settings where p_model represents a pretrained model
        and this prior constrains the new model relative to it.
        """
        log_prior=0.0
        s=self.std
        for p,m in zip(model.parameters(),p_model.parameters()):
            log_prior+=torch.sum(-0.5*((p-m)/s)**2-0.5*torch.log(2*torch.tensor(3.1415926)*s**2))*scaling

        return log_prior      
    
class HyperPrior(nn.Module):
    
    def __init__(self,DataLoader,rescale,prior):
        super(HyperPrior,self).__init__()
        self.dl=DataLoader
        self.rescale=rescale
        self.prior=prior

    def forward(self,model,scale):
        """
        Empirical Bayes hyperprior combining prior and data likelihood of pre-training dataset.
        
        Used for learning prior hyperparameters. Computes:
        log p(θ|data) ∝ log p(θ) + log p(data|θ)
        """
        dataset_size=self.dl.len()
        effective_size=dataset_size*self.rescale
        lp1=self.prior(model,1/effective_size)
        
        X_batch,Y_batch=self.dl.sample()
        lp2=model.evaluate((X_batch,Y_batch))
        
        return (lp1+lp2)*(scale*effective_size)

class TransferLearningHyperPrior(nn.Module):
    
    def __init__(self,prior,hyper_prior):
        super(TransferLearningHyperPrior,self).__init__()
        self.prior=prior             #prior for the actual model e.g. gaussian mean field with p_model as a mean
        self.hprior=hyper_prior      #prior for the p_model e.g. posterior of the p_model on the auxiliary dataset
        
    def forward(self,models,scale):
        """
        Hierarchical transfer learning prior.
        
        Combines two priors:
        - prior: Gaussian prior on current model relative to pretrained model
        - hprior: Empirical Bayes prior on the pretrained model itself
        
        Args:
            models: Object with .model and .p_model attributes
            scale: Prior strength scaling factor
        """
        model=models.model
        p_model=models.p_model
        lp1=self.prior(model,p_model,scale)
        lp2=self.hprior(p_model,scale)
        
        return lp1+lp2
        
        
        
        
        
        
        
        
        