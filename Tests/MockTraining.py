"""Mock training process for testing process communication.
Follows the communication protocol of Training.py, except no models are build or trained and such 
requests are simply replace with sleep statements."""

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
    sys.path.insert(0, path)

    from OTFFineTune.TrainProc import SetTrainProcStatus,GetTrainProcStatus
    builder_args=[float(arg) for arg in sys.argv[7:]]
    models=[]
    if init_type=='R':
        if builder_func=='SpiceNequIP':
            from OTFFineTune.SpiceModelLoader import NequIP_Builder
        elif builder_func=='MACE':
            from OTFFineTune.MACE_Loader import MACE_Builder
        for i in range(n_models):
            time.sleep(0.1)  #simulate load time          

    else:

    
        if builder_func=='SpiceNequIP':
            from OTFFineTune.SpiceModelLoader import NequIP_Builder
            for i in range(n_models):
                time.sleep(0.1)  #simulate build time
        elif builder_func=='MACE':
            from OTFFineTune.MACE_Loader import MACE_Builder
            for i in range(n_models):
                time.sleep(0.1)  #simulate build time

    SetTrainProcStatus(pid,'Finished')
    done=False
    while not done:
        time.sleep(1)
        status=GetTrainProcStatus(pid)
        if status=="Shutdown":
            i=0
            time.sleep(0.1)  #simulate shutdown time
            SetTrainProcStatus(pid,'Shutting Down')
            done=True
            break
        if status=="Training Request":
            SetTrainProcStatus(pid,'Training')
            new_data=torch.load('tmp/new_data')
            i=0
            for model in models:
                for cycle in range(1):
                    time.sleep(0.1)  #simulate training time
                print('test', pid, i)
                torch.save(model.state_dict(),'model_dict{}{}'.format(pid,i))
                torch.save(model,'Checkpoints/model{}{}'.format(pid,i))
                i+=1
            SetTrainProcStatus(pid,'Finished')