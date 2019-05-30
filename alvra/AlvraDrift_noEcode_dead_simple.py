from bsread import source
import numpy as np
import epics
import os
import photodiag
from collections import deque
from simple_pid import PID
import datetime
import time
import shutil

Drift_Corr_Master = True 


# Define channels to read
I0_PV = "SAROP11-CVME-PBPS2:Lnk9Ch15-DATA-SUM"
PALM_DELAY_PV = "SAROP11-PALMK118:ANALYSIS_PALM_DEL"

DRIFT_CORR_PV = epics.PV('SAROP11-PALMK118:DRIFT_CORR_ON')


calibBAM = 1000
mm2fs = 6666.6
fs2mm = 1/mm2fs


STAGE_PV = epics.PV('SLAAR11-LMOT-M452:MOTOR_1.VAL')

# little utilities functions
def now(): return str(datetime.datetime.now())



def myprint(*args,end="\n"):
    args = [str(arg) for arg in args]
    string = "%26s"%now() + " " + " ".join(args)
    ncolumns = shutil.get_terminal_size().columns

    npadding = ncolumns - len(string)
    fmt = "%%%ds"%npadding
    print(string + fmt%" ",end=end)
    

class Adeque(deque):
    def _init_(self, *args, **kwargs):
        super()._init_(*args, **kwargs)

    @property
    def mean(self):
        if len(self) == 0:
            return 0
        else:
            return sum(self)/len(self)

class Data(object):
    def __init__(self,pvname,running_average_window=30):
        self.pvname = pvname
        self.pv = epics.PV(pvname)
        self.queue = Adeque(maxlen=running_average_window)

    def get(self): return self.pv.get()

    def put(self,value): return selv.pv.put(value)

    @property
    def mean(self): return self.queue.mean

    def append(self,value=None):
        if value is None: value = self.get()
        self.queue.append(value)

I0 = Data(I0_PV)
PALM_DELAY = Data(PALM_DELAY_PV,running_average_window=1000)


# TODO: put the following params into the panel
PALM_DELAY_TARGET = 0 
PALM_DBand = 10 #was 17... 

StageMovLimits = 200



# ---- Start processing
n = 1
current_position = STAGE_PV.get()
while True:
    storage = dict()
    i0 = I0.get()
    while i0 < 0.1:
        i0 = I0.get()
        time.sleep(0.2)
        myprint("waiting for beam ...",end="\r")

    delay = PALM_DELAY.get()
    PALM_DELAY.append(delay)
    time.sleep(0.05)



    # Feedback stage of Global time laser
    delta_t = PALM_DELAY_TARGET-PALM_DELAY.mean
    correction = delta_t*fs2mm
    if n%400 == 0: current_position = STAGE_PV.get()
    #if n%1==0:
    myprint("%10d, last delay %+6.1f fs, running avg delay %+6.1f fs, target delay %+6.1f fs, correction %+7.4f mm, stage current position %+7.4f mm"%
         (n,delay,PALM_DELAY.mean,PALM_DELAY_TARGET,correction,current_position), end="\r")

    n += 1
    if (n%1000) != 0: continue


    correction = (PALM_DELAY_TARGET-PALM_DELAY.mean)*fs2mm
    new_position = STAGE_PV.get()+correction

    b_enough_data = len(PALM_DELAY.queue) > 10 # in case there is no beam when we start
    b_dband = abs(PALM_DELAY_TARGET-PALM_DELAY.mean)> PALM_DBand
    b_limits = abs(new_position) < StageMovLimits
    b_is_correction_on = DRIFT_CORR_PV.get() == 1
    
    if b_enough_data and b_dband and b_limits and b_is_correction_on:
        myprint("moving the stage to %7.4f mm"%new_position)
        STAGE_PV.put(new_position)
        PALM_DELAY.queue.clear()
        time.sleep(0.5)
        current_position = STAGE_PV.get()
        
