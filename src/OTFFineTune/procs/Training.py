"""
Training Subprocess Manager

This module runs as a separate subprocess spawned by EnsembleFF. Each subprocess:
1. Initializes an ensemble of neural network models on a specific GPU
2. Listens for training requests from the main process
3. Executes SGHMC optimization cycles on new data
4. Saves model checkpoints after training

The subprocess communicates with the main process through shared temporary files.
Model state is serialized to CPU (due to HPC memory layout constraints) for
inter-process transfer.

Command-line arguments:
    pid: Process ID (0 to nprocs-1)
    target_dev: CUDA device index
    n_models: Number of models this process manages
    builder_func: Model class ('SpiceNequIP' or 'MACE')
    init_type: 'I' (initialize new) or 'R' (restore from checkpoint)
    path: Path to code repository
    constructor_args: Variable arguments for model builder
"""

"""
Training Subprocess for Ensemble Model Updates

This module runs as a separate process (one per GPU device) and handles the
continuous retraining of neural network potentials using SGHMC optimization.

Workflow:
1. Initialize ensemble of models on assigned GPU device
2. Load model checkpoints if resuming (restart=True)
3. Poll for training requests from main process
4. On request: load new data, update each model with SGHMC cycles
5. Save updated model states and signal completion

The subprocess uses file-based communication via status files in tmp/ for
compatibility with HPC environments and checkpoint/restart capabilities.

Entry Point:
    python Training.py <pid> <device_id> <n_models> <builder> <init_type> <path> [builder_args...]
    
    Args:
        pid: Process ID for this training subprocess
        device_id: CUDA device ID
        n_models: Number of models in ensemble
        builder: Model builder name ('SpiceNequIP' or 'MACE')
        init_type: 'I' for fresh initialization, 'R' for restart from checkpoint
        path: Path to code repository
        builder_args: Additional arguments for model builder
"""

import sys

import os
import time
import torch
import yaml
import sys







if __name__ == "__main__":



        #(target_dev,pid,n_models,builder_func)=sys.argv[1:5]
    (pid,target_dev,n_models,builder_func,init_type,path)=sys.argv[1:7]
    #for build testing, target_dev will be cpu
    if target_dev=='cpu':
        target_dev=torch.device('cpu')
    else:
        target_dev=torch.device("cuda:{}".format(target_dev))
    pid=int(pid)
    n_models=int(n_models)

    file_path=os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path)
    from comm.TrainProc import SetTrainProcStatus,GetTrainProcStatus
    builder_args=[float(arg) for arg in sys.argv[7:]]
    models=[]
    if init_type=='R':
        if builder_func=='SpiceNequIP':
            from OTFFineTune.models.SpiceModelLoader import NequIP_Builder
        elif builder_func=='MACE':
            from OTFFineTune.models.MACE_Loader import MACE_Builder
        for i in range(n_models):
            model=torch.load('Checkpoints/model{}{}'.format(pid,i),map_location=torch.device('cpu'),weights_only=False)
            model=model.to(target_dev)
            model.change_device(target_dev)
            models.append(model)            
    else:

    
        if builder_func=='SpiceNequIP':
            from OTFFineTune.models.SpiceModelLoader import NequIP_Builder
            for i in range(n_models):
                model=NequIP_Builder(builder_args).to(target_dev)
                model.change_device(target_dev)
                models.append(model)
        elif builder_func=='MACE':
            from OTFFineTune.models.MACE_Loader import MACE_Builder
            for i in range(n_models):
                model=MACE_Builder(builder_args).to(target_dev)
                model.change_device(target_dev)
                models.append(model)

    SetTrainProcStatus(pid,'Finished')
    done=False
    while not done:
        time.sleep(1)
        status=GetTrainProcStatus(pid)
        if status=="Shutdown":
            i=0
            for model in models:
                torch.save(model,'model_dict{}{}'.format(pid,i))
                i+=1
            SetTrainProcStatus(pid,'Shutting Down')
            done=True
            break
        if status=="Training Request":
            SetTrainProcStatus(pid,'Training')
            new_data=torch.load('tmp/new_data',weights_only=False)
            i=0
            for model in models:
                for cycle in range(1):
                    model.update(new_data)
                   # fp = open('tmp/training{}.log'.format(pid))
                    #lines=fp.readlines()
                    #e,f,s,u=lines[-2].split(' ')
                    #e,f,s,u=float(e),float(f),float(s),float(u)
                    #if e<0.1 and f<0.2 and u<0.1:
                     #   break
                    #else:
                     #   print('convergence not reached after {} cycles'.format(cycle))

                print('test', pid, i)
                torch.save(model.state_dict(),'model_dict{}{}'.format(pid,i))
                torch.save(model,'Checkpoints/model{}{}'.format(pid,i))
                i+=1
            SetTrainProcStatus(pid,'Finished')


