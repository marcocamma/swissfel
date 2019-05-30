import numpy as np
import glob
import os
import h5py
import sys
sys.path.append("/afs/psi.ch/sys/psi.ra/Programming/psi-python36/5.0.1/lib/python3.6/site-packages/")
import trx
import re
import matplotlib.pyplot as plt
import joblib
import time



g_folder = "/sf/bernina/config/com/data/scan_info"




_timeInStr_regex = re.compile(r'_(-?\d+\.?\d*(?:fs|ps|ns|us|ms|s))', re.UNICODE)



def getDelayFromString(string) :
    match = _timeInStr_regex.search(string)
    return match and match.group(1) or None


def mkJson(searchstr):
    """ for data saved on wed. night (tiox/l03/run*) """
    files = glob.glob(searchstr)

    # take only data for which on and off exists
    #files = [ f for f in files if f.replace("off","on") in 
    if len(files) == 0:
        print("No files found in",folder)
        return
    # exctract delay from filename
    delays = [trx.utils.strToTime(getDelayFromString(fname)) for fname in files]
    delays = np.asarray(delays)
    #sort based on delay
    idx = np.argsort(delays)
    files = np.asarray(files)
    files = files[idx]
    delays = delays[idx]
    o = {}
    o['scan_files'] = [ [tf] for tf in files]
    o['scan_parameters'] = {
            'Id': ['SLAAR01-TSPL-EPL'],
            'name':['lxt']
            }
    o['scan_readbacks'] = [ [td] for td in delays]
    o['scan_values'] = [ [td] for td in delays]
    o['scan_step_info'] = [None for td in delays]


    return o




def readJson(n=1):
    """ for eco scan files """
    import json
    import glob
    files = glob.glob(g_folder + "/scan%03d*"%n)
    fname = files[0]
    with open(fname) as f:
      a = json.load(f)
    return datastorage.DataStorage(a)

def _findFiles(folder):
    files = []
    for runfolder in glob.iglob(folder + "/*"):
        _temp = glob.glob( runfolder + "/*.h5")
        files.extend(_temp)
    return files


