import yaml


import Procs


def StartupSequence(Config):

    cfg=yaml.full_load(open(Config))
    Procs.ProcComSetUp()

    #Setting up DFT process and its ReqHandler

    if cfg['DFTResourceManager']=='SLURM':

        #Add error handling
        DFTProcSubmitFile=cfg['DFTProcSubmitFile']

        if cfg['DFTCode']=='vasp_std':
            DFTReqHandler=Procs.VASPSLURMBuilder(DFTProcSubmitFile)

    
    if cfg['NNPResourceManager']=='SLURM':

        NNPProcSubmitFile=cfg['NNPProcSubmitFile']
        NNP=cfg['NNP']
        ReqHandler=Procs.NNPSLURMBuilder(NNP,NNPProcSubmitFile)
        

    return ReqHandler