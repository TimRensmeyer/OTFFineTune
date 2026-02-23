# OTFFineTune: On-the-Fly Finetuning Workflow


```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                      ASE Molecular Dynamics Driver                          │
│                         (MD, BFGS, Langevin, etc.)                          │
│                                                                             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             │ ase.Atoms object
                             v
                  ┌──────────────────────┐
                  │  FileIOReqHandler    │
                  │  (write atoms.xyz)   │
                  └──────────────────────┘
                             │
                             v
           ┌─────────────────────────────────────────┐
           │             MLFFProc.py                 │
           │                                         │
           │  ┌───────────────────────────────────┐  │
           │  │  EnsembleFF (N models on GPUs)    │  │
           │  │  └─ Model 1, 2, 3, ... N          │  │
           │  │  └─ Each predicts: E, F, σ_E, σ_F │  │
           │  └───────────────────────────────────┘  │
           │             ↓                           │
           │  ┌───────────────────────────────────┐  │
           │  │  OTFForceField                    │  │
           │  │  (decision logic)                 │  │
           │  │                                   │  │
           │  │  1. Combine ensemble predictions  │  │
           │  │  2. Compute confidence metric     │  │
           │  │  3. Evaluate: conf >= 0.95 ?      │  │
           │  └───────────────────────────────────┘  │
           │             │                           │
           └─────────────┼───────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              v YES                 v NO / STEP 1
         ┌─────────────┐       ┌──────────────────┐
         │ Return ML   │       │ Request DFT      │
         │ Predictions │       │ Calculation      │
         └──────┬──────┘       └────────┬─────────┘
                │                       │
                │              ┌────────v────────┐
                │              │  FileIOReqHandler
                │              │  (write POSCAR) │
                │              └────────┬────────┘
                │                       │
                │              ┌────────v─────────────┐
                │              │  VASPProc.py         │
                │              │  (DFT Subprocess)    │
                │              │  srun vasp_std       │
                │              └────────┬─────────────┘
                │                       │
                │              ┌────────v─────────────┐
                │              │  FileIOReqHandler    │
                │              │  (read OUTCAR)       │
                │              └────────┬─────────────┘
                │                       │
                │              ┌────────v────────────────┐
                │              │  Recalibrate Uncertainty│
                │              │  Update error metrics   │
                │              └────────┬────────────────┘
                │                       │
                │              ┌────────v─────────────────┐
                │              │  Trigger Parallel        │
                │              │  Training.py Resampling  │
                │              └────────┬─────────────────┘
                │                       │
                │              ┌────────v────────────────┐
                │              │ Return DFT Predictions  │
                │              └────────┬────────────────┘
                │                       │
                └───────────────┬───────┘
                                │
                    ┌───────────v────────────┐
                    │ FileIOReqHandler       │
                    │ (return .npy files)    │
                    └───────────┬────────────┘
                                │
                                v
                    ┌───────────────────────┐
                    │ MD Driver Receives    │
                    │ Energy, Forces,       │
                    │ (Stress), Uncertainties
                    │ (zeros if DFT used)   │
                    └───────────┬───────────┘
                                │
                                v
                    ┌───────────────────────┐
                    │ Next MD Step Computed │
                    │ Cycle repeats...      │
                    └───────────────────────┘
```

---



# Startup & Initialization

**Entry Point**: A Startup Manager (currently only OTFFineTune.procs.comm.Procs.SlurmStartup implemented) Launches the MLFFProc.py and VASPProc.py
 in a SLURM job on a node of an HPC Cluster and returns a Request Handler callable, which takes ase.Atoms instances as input and returns predictions.

```
Startup Manager (SlurmStartup or direct)
    │
    ├─ [1] Spawn MLFFProc.py subprocess on HPC compute node
    │       │
    │       ├─ Load runconfig.yaml (model type, GPU list, ensemble size, thresholds)
    │       │
    │       ├─ [2] Spawn VASPProc.py subprocess  (This process is launched by the Startup Manager itself in the dev branch and this will become the standard in the 
    |       |                                     main branch too in the future, to allow the VASPProc.py to run on different hardware from the ML parts of the code)
    │       │
    │       ├─ [3] Initialize EnsembleFF with N models loaded on specified GPUs
    │       │       └─ Each model: weights on separate GPU device
    │       │
    │       ├─ [4] Create OTFForceField instance (main orchestrator in GPU process)
    │       │
    │       ├─ [5] Spawn N instances of Training.py (GPU subprocesses, one per device)
    │       │       │
    │       │       └─ Each Training.py polls 'train{i}_status.txt' for work requests
    │       │           Status: "Idle" → "Training Request" → "Training" → "Finished"
    │       │
    │       └─ [6] Wait for all Training.py processes to signal status = "Finished"
    │           (indicates ready state before accepting MD requests)
    │
    ├─ [7] Create DirectoryStructure
    │       │
    │       ├─ ./tmp/           # Inter-process communication (atoms.xyz, new_data, status files)
    │       │
    │       ├─ ./Coords/        # Log of all atomic structures processed
    │       ├─ ./ML_preds/      # ML-predicted results
    │       ├─ ./DFT_preds/     # VASP-calculated reference results
    │       └─ ./Checkpoints/   # Model state_dicts for restart
    │
    └─ [8] Set gpu_status.txt = "OTF Force Field Ready"
           └─ Signal to MD driver that system is ready for requests
```

**Practical Considerations**: 
In practice a small wrapper can be implemented that turns the RequestHandler into an ase calculator, which is done for example done in the
run.py files in the Tutorials. The current Startup Manager Launches the processes in a seperate SLURM job while the Request Handler lives on 
the login node. 

This was done among other things, so that you can use the OTFForceField interactively within jupyter notebook files.
Because in the future I want to extend the code so that DFT calculations and ML training/inference don't have to 
happen within the same SLURM job and to enable the interactive jupyter notebook integration, the process communication is implemented 
in a primitive way via text files in the tmp directory, since communication of processes on SLURM jobs with processes outside of that job
is often restricted on hpc clusters but file read/write should always be possible. 
Note that using seperate hpc nodes for DFT and ML parts of the code already exists as a functionality in the dev branch and will be merged into the main branch soon.



