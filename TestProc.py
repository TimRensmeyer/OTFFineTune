from Procs import SetProcStatus, GetProcStatus
import time




if __name__ == "__main__":

    done=False
    while not done:
        status=GetProcStatus()
        if status =='Calculation Request':
            SetProcStatus('Calculating')
            time.sleep(20)
            SetProcStatus('Finished Calculating')
        elif status=='Shutdown':
            done=True
            break
        else:
            time.sleep(1)

