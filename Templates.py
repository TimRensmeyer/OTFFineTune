#!/usr/bin/env python
# coding: utf-8

# In[50]:


import torch
import torch.nn as nn
import abc
from abc import abstractmethod

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

# the base class for monte carlo markov chain based optimizers that all such optimizer classes should
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
