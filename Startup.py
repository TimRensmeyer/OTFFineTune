"""
Startup Configuration and Initialization

This module provides utilities for initializing the OTF fine-tuning workflow.
It reads YAML configuration and sets up DFT and NNP request handlers with
appropriate resource managers (SLURM, local, etc.).

Note: Most functionality has been superseded by direct integration in MLFFProc.py.
This module is retained for compatibility but may be deprecated.
"""

import yaml


import Procs


def StartupSequence(Config):
    """
    Parse configuration and initialize process handlers.
    
    Note: This function is currently no longer used and may be removed in future versions.
    Startup logic has been integrated into MLFFProc.py.
    
    Args:
        Config: Path to YAML configuration file
        
    Returns:
        DFT request handler callable
    """

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