import glob
import os
import trx
import json

def find_last_pedestal(folder="/sf/bernina/config/com/data/raw/"):
  files=list(glob.glob( os.path.join(folder,"pedestal_*")) )
  files.sort()
  return files[-1]


def run_to_scan(folder,scan_number=1):
    folder = folder.rstrip("/"); 
    files=list(glob.glob( os.path.join(folder,"*_rep0.h5")) )
    files.sort()
    res = dict( scan_values = [], scan_files = [] )
    for f in files:
        basename = os.path.basename(f)
        delay = trx.utils.getDelayFromString(basename)
        delay = trx.utils.strToTime(delay)
        res["scan_files"].append( [f,""] )
        res["scan_values"].append( [delay,] )
    name = folder.split("/")[-3:]
    name = "_".join(name)
    json_name = "../../scan_info/scan%03d_%s_scan_info.json"%(scan_number,name)
    print("Will save scan info in %s"%json_name)
    if not os.path.isfile(json_name):
        with open(json_name,"w") as f: json.dump(res,f)
    else:
        print("file exists")
    return res
