import subprocess
import torch
import torch.nn as nn
import numpy as np
import os

def TrainProcComSetUp(nprocs):
    for i in range(nprocs):
        fp=open('./tmp/train{}_status.txt'.format(i), 'w')
        fp.write('ready')
        fp.close()

def SetTrainProcStatus(i,status):
    fp=open('./tmp/train{}_status.txt'.format(i), 'w')
    fp.write(status)
    fp.close()

def GetTrainProcStatus(i):
    fp=open('./tmp/train{}_status.txt'.format(i), 'r')
    status=fp.read()
    fp.close()
    return status

def SetTrainRequest(nprocs):
    for i in range(nprocs):
        SetTrainProcStatus(i,'Training Request')

def GetTrainStatus(nprocs):
    status='Finished'
    for i in range(nprocs):
        if GetTrainProcStatus(i) != 'Finished':
            status='running'
            break
    
    return status