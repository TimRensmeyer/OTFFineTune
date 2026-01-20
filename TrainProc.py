"""
Training Process Communication Interface

Provides utilities for managing inter-process communication with training subprocesses.
Each training subprocess maintains a status file that the main process can query
and update to coordinate training requests and synchronization.

Status Files (in tmp/):
- train{i}_status.txt: Status of training process i

Status Values:
- 'ready': Subprocess initialized
- 'Training Request': Main process requesting model update
- 'Training': Subprocess actively retraining
- 'Finished': Subprocess completed training cycle
- 'Shutdown': Shutting down subprocess
- 'Shutting Down': Subprocess cleanup in progress
"""

import subprocess
import torch
import torch.nn as nn
import numpy as np
import os

def TrainProcComSetUp(nprocs):
    """Initialize status files for nprocs training subprocesses."""
    for i in range(nprocs):
        fp=open('./tmp/train{}_status.txt'.format(i), 'w')
        fp.write('ready')
        fp.close()

def SetTrainProcStatus(i,status):
    """Set status of training process i."""
    fp=open('./tmp/train{}_status.txt'.format(i), 'w')
    fp.write(status)
    fp.close()

def GetTrainProcStatus(i):
    """Get current status of training process i."""
    fp=open('./tmp/train{}_status.txt'.format(i), 'r')
    status=fp.read()
    fp.close()
    return status

def SetTrainRequest(nprocs):
    """Signal all training processes to start retraining cycle."""
    for i in range(nprocs):
        SetTrainProcStatus(i,'Training Request')

def GetTrainStatus(nprocs):
    """
    Check if all training processes have finished current cycle.
    
    Returns:
        'Finished' if all processes are finished
        'running' if any process is still training
    """
    status='Finished'
    for i in range(nprocs):
        if GetTrainProcStatus(i) != 'Finished':
            status='running'
            break
    
    return status