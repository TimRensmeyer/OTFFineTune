"""This Training Process module runs just like the full training process,
except that it only performs two optimization steps per training cycle"""

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
    #sys.path.insert(0, path)
    sys.path.insert(0, os.path.join(path, 'OTFFineTune'))

    from OTFFineTune.procs.comm.TrainProc import SetTrainProcStatus,GetTrainProcStatus
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
                model=NequIP_Builder(builder_args,testing=True).to(target_dev)
                model.change_device(target_dev)
                models.append(model)
        elif builder_func=='MACE':
            from OTFFineTune.models.MACE_Loader import MACE_Builder
            for i in range(n_models):
                model=MACE_Builder(builder_args,testing=True).to(target_dev)
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

                print('test', pid, i)
                torch.save(model.state_dict(),'model_dict{}{}'.format(pid,i))
                torch.save(model,'Checkpoints/model{}{}'.format(pid,i))
                i+=1
            SetTrainProcStatus(pid,'Finished')


