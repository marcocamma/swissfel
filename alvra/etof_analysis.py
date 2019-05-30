import numpy as np
import h5py
import photodiag
import time
import glob
import datetime
import pathlib
import os
import tqdm
from matplotlib import pyplot as plt
try:
    from datastorage import DataStorage
except ImportError:
    DataStorage = dict


# PALM data names
streaked = 'SAROP11-PALMK118:CH1_BUFFER'
reference = 'SAROP11-PALMK118:CH2_BUFFER'

data_folder = "/sf/alvra/data/p17807/raw/"

energy_range = np.linspace(2000,2400,1000)

palm = photodiag.PalmSetup(
    channels={'0': reference, '1': streaked},
    noise_range=[0, 250], energy_range=energy_range,
)

calib_file = '/sf/photo/src/PALM/calib/2019-05-24_13:40:42.palm_etof'
palm.load_etof_calib(calib_file)

calibPALM = -2.71*1e-15

offsets = dict(
    pump_delay_stage = 146.625-0.07,
    common_delay_stage = 24.998
)


tosave = dict(
    laser_opa_diode  ="SLAAR11-LSCP1-FNS:CH2:VAL_GET",
    pump_delay_stage ="SLAAR11-LMOT-M451:ENC_1_BS",
    common_delay_stage = "SLAAR11-LMOT-M452:ENC_1_BS",
)


testfile=pathlib.Path("/sf/alvra/data/p17807/raw/10ps_1.2uJ_l_MbCO/run_001075.BSREAD.h5")

def process_etof(data,Nshots=None):
    ref_vals = data[reference+"/data"]
    str_vals = data[streaked+"/data"]
    delay, pulse_length = palm.process(
        {'0': -ref_vals[:Nshots], '1': -str_vals[:Nshots]}, noise_thr=0, jacobian=False, peak='max',
    )
    delay = np.round(delay*calibPALM,15)
    pulse_id = data[reference+"/pulse_id"][:Nshots]
    return dict(etof_delay=delay,pulse_id=pulse_id)

def _diodename(link,channel):
    return "data/SAROP11-CVME-PBPS2:Lnk%dCh%d-DATA-SUM/data"%(link,channel)

def get_diodes(data,link=9,channels=(12,13,14,15),Nshots=None):
    ret = dict()
    names = ["xray_i0_lnk%dch%d" % (link,channel) for channel in channels]
    diodes = [data[_diodename(link,channel)][:Nshots] for channel in channels]

    for name,diode in zip(names,diodes): ret[name]=diode
    ret["xray_i0_sum"] = np.sum(np.asarray(diodes),axis=0)
    
    for key in ret: ret[key]=np.squeeze(ret[key])

    return ret


   

def process_data(data,Nshots=None,return_all=False,i0= {'link' : 9, 'channels' : (12,13,14,15)} ):
    try:
        ret = process_etof(data["data"],Nshots=Nshots)
    except KeyError:
        print(" ... could not find etof ... ",end="")
        ret = dict()

    for key,name in tosave.items():
        try:
            ret[key] = data["data/"+name+"/data"][:Nshots]
            #if "pulse_id" not in ret: ret["pulse_id"] = data["data/"+name+"/pulse_id"]
        except Exception as e:
            print("could not read",key)

    keys = list(ret.keys())
    for key in keys:
        if key.endswith("_stage"):
            timename = key[:-5] + "time"
            ret[timename] = (ret[key][:Nshots]-offsets[key])*2*1e-3/3e8
            ret[timename] = np.round(ret[timename],15)


    ret.update( get_diodes(data,Nshots=Nshots,**i0) )

    for key,value in ret.items(): ret[key] = np.squeeze(ret[key])


    if not return_all:
        idx = ret["pulse_id"]%2==0
        for key in ret: ret[key] = ret[key][idx]

    return ret


def do_file(fname,Nshots=None,save=False,force=False):

    fdelays_name = pathlib.Path(fname).stem
    fdelays_name = "delays/"+fdelays_name+".delays.h5"

    if pathlib.Path(fdelays_name).is_file() and not force:
        ret = dict( h5py.File(fdelays_name,"r") )
        try:
            for key in ret: ret[key] = ret[key].value 
        except ValueError:
            pass
    else:
        data = h5py.File(fname,"r")
        if "data" not in data: raise RuntimeError("No data, must be a messed up file")
        ret = process_data(data,Nshots=Nshots)
        ret["info"] = "C. Arrell scripts usually subtract the average laseroff ETOF to the ETOF to correct for ETOF drifts, this is NOT done by this script"
        for key in offsets: ret["offset_%s"%key] = offsets[key]
        if save:
            with h5py.File(fdelays_name,"w") as fout:
                for key,value in ret.items(): fout[key]=value
    return ret


def do_all_may_files(force=False,raise_exception=False):
    folders = glob.glob(data_folder,recursive=True)
    beginning_of_may_exp = datetime.datetime(2019,5,15).timestamp()
    folders = [f for f in folders if os.stat(f).st_ctime > beginning_of_may_exp]
    folders = sorted(folders)
    files = []
    for f in folders:
        files.extend( glob.glob(f+"/*BSREAD.h5") )
    bar = tqdm.tqdm(total=len(files))
    files = sorted(files,key=os.path.basename)
    for ifile,fname in enumerate(files):
        basename = str(pathlib.Path(fname).stem)
        bar.set_description(basename)
        #print("  ->",pathlib.Path(fname).name,"...",end="")
        try:
            do_file(fname,save=True,force=force)
            #print("...done")
        except Exception as e:
            if raise_exception: raise(e)
            print(basename,"...failed, error '%s'"%e)
        bar.update(ifile)

def read_delay_file(fname):
    h = h5py.File(fname,"r")
    data = dict()
    for key in h: data[key]=h[key]
    for key,value in data.items():
        try:
            data[key] = value.value
        except:
            pass
    h.close()
    return data

def read_delay_files(runs = range(1271,1290) ):
    data = []
    for run in runs:
        try:
            data.append(read_delay_file("delays/run_%06d.BSREAD.delays.h5"%run))
        except:
            print("Could not read run",run)
    ret = {}
    for key in data[0].keys():
        try:
            ret[key] = np.hstack( [d[key] for d in data] )
        except ValueError:
            ret[key] = data[0][key] # for info
    return ret

def running_average(y,every=50):
    kernel = np.ones(every)/every
    return np.convolve(y,kernel,mode="same")

    
def plot_data_analysis(data_or_filename,nshot=6):
    if type(data_or_filename) not in (dict,DataStorage):
        data = read_delay_file(data_or_filename)
    else:
        data = data_or_filename
    fig,ax=plt.subplots(3,2,sharey="row",sharex="col",gridspec_kw=dict(width_ratios=[0.75,0.25]))
    pulse_id = data["pulse_id"][:]
    i0 = data["xray_i0_sum"][:]
    xon = i0>0.3
    loff = data["laser_opa_diode"][:] < 100
    thz_delay = data["etof_delay"][:]*1e12

    ax[0,0].plot(pulse_id[~loff],thz_delay[~loff],".",label="THz laser on")
    ax[0,0].plot(pulse_id[loff],thz_delay[loff],".",label="THz laser off",ms=5)
    ax[0,0].plot(pulse_id[~loff],running_average(thz_delay[~loff],300),color="0.2",label="THz laser on (300 av)")
    ax[0,0].plot(pulse_id,data["common_delay_time"]*1e12,label="globi position")

    jitter_lon = np.std(thz_delay[~loff & xon])*1e3
    jitter_loff = np.std(thz_delay[loff & xon])*1e3

    h1,b = np.histogram(thz_delay[~loff&xon],bins=np.arange(-0.3,0.3,0.005))
    h2,b = np.histogram(thz_delay[loff&xon],bins=np.arange(-0.3,0.3,0.005))
    ax[0,0].set_ylim(-0.15,0.2)
    b = (b[:-1]+b[1:])/2
    ax[0,1].plot(h1,b,label="xon, lon, std %.1f"%jitter_lon)
    ax[0,1].plot(h2,b,label="xon, loff, std %.1f"%jitter_loff)
    ax[0,1].legend()
    ax[0,0].set_xlabel("pulse_id")
    ax[0,0].set_ylabel(r"arrival time (ps)")
    ax[0,1].set_xlabel("histogram")
    #ax[0].set_title(pathlib.Path(fname).name)
    ax[0,0].legend()

    # X-ray i0
    ax[1,0].plot(pulse_id,i0,".",label="i0")
    ax[1,0].set_ylabel(r"X-ray I0")
    h1,b = np.histogram(i0,bins=np.arange(-0.1,1.7,0.03))
    b = (b[:-1]+b[1:])/2

    xnoise = np.std(i0[xon])
    xmean = np.mean(i0[xon])
    ax[1,1].plot(h1,b,label="RMS noise (xon) %.2f%%"%(xnoise/xmean*100))
    ax[1,1].set_ylim(-0.15,2.0)
    ax[1,1].legend()


    ax[2,0].plot(pulse_id,data["laser_opa_diode"],".",label="i0")
    ax[2,0].set_ylabel(r"laser_opa_diode")
    lnoise = np.std(data["laser_opa_diode"][~loff])
    lmean = np.mean(data["laser_opa_diode"][~loff])
    h1,b = np.histogram(data["laser_opa_diode"],bins=np.arange(-100,1000,5))
    b = (b[:-1]+b[1:])/2
    ax[2,1].plot(h1,b,label="RMS noise (lon) %.2f%%"%(lnoise/lmean*100))
    ax[2,1].legend()
    for a in ax.ravel(): a.grid()

def do150fs():
    data = read_delay_files(range(1270,1312))
    plot_data_analysis(data)
    return data

def do600fs():
    data = read_delay_files(range(1312,1332))
    plot_data_analysis(data)
    return data

def do450fs():
    data = read_delay_files(range(1335,1361))
    plot_data_analysis(data)
    return data

def do5ps():
    runs = list(range(1361,1374))+list(range(1406,1414))
    data = read_delay_files(runs)
    plot_data_analysis(data)
    return data

def do1_3ps():
    data = read_delay_files(range(1374,1406))
    plot_data_analysis(data)
    return data

def do450fs4uJ():
    data = read_delay_files(range(1414,1432))
    plot_data_analysis(data)
    return data

def do450fs2():
    data = read_delay_files(range(1432,1451))
    plot_data_analysis(data)
    return data

def do450fs3():
    data = read_delay_files(range(1451,1492))
    plot_data_analysis(data)
    return data

def do150fs2():
    data = read_delay_files(range(1492,1520))
    plot_data_analysis(data)
    return data

def do300fs():
    data = read_delay_files(range(1520,1556))
    plot_data_analysis(data)
    return data

def do750fs():
    data = read_delay_files(range(1556,1586))
    plot_data_analysis(data)
    return data

def do600fs():
    data = read_delay_files(range(1586,1612))
    plot_data_analysis(data)
    return data

def do900fs():
    data = read_delay_files(range(1612,1638))
    plot_data_analysis(data)
    return data






#if __name__ == "__main__":
#    do_all_may_files()

