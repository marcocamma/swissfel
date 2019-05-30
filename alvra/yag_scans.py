import numpy as np
from matplotlib import pyplot as plt
import pathlib
import json
import joblib
import h5py
import lmfit
import os
from scipy.special import erf

from ..utils import subtractReferences
from . import etof_analysis

sqrt2=np.sqrt(2)


cache = joblib.Memory(location="./cache",verbose=0)

json_folder = pathlib.Path("/sf/alvra/data/p17808/res/scan_info/")


def read_json(fname="YAGSample_FEL0.3_2705_100mbar_026_scan_info.json"):
    if not os.path.exists(fname):
        fname = json_folder / fname
    with open(fname,"r") as f: info = json.load(f)
    scan_motor = info["scan_parameters"]["Id"][0]
    scan_readbacks   = [i[0] for i in info["scan_readbacks"]] 
    scan_files = [i[0] for i in info["scan_files"]]
    return dict(scan_readbacks=scan_readbacks,scan_motor=scan_motor,scan_files=scan_files)


@cache.cache
def ana_h5(fname,detector="SLAAR11-LSCP1-FNS:CH0:VAL_GET"):
    h = h5py.File(fname,"r")
    diode = h[detector]
    pulse_id = diode["pulse_id"][:]
    data = diode["data"][:]
    idx_ref = np.argwhere(pulse_id % 2 == 1).ravel()
    data_norm = subtractReferences(data,idx_ref, useRatio = True)
    xon = pulse_id % 2 == 0
    #diode_xray_off = np.sum(data[~xon])
    diode_xray_on = data_norm[xon]
    try:
        etof = etof_analysis.process_etof(h)["etof_delay"][xon]
    except:
        etof = None
    #norm_diff = (diode_xray_on-diode_xray_off)/diode_xray_off
    return dict(mean=diode_xray_on.mean(),median=np.median(diode_xray_on),
        diode = diode_xray_on,etof=etof)
    

def conv_gauss_and_const(x,sig):
    return 0.5*(1-erf(-x/sqrt2/sig))

def conv_gauss_and_exp(x,sig,tau):
    return 0.5*np.exp(-(2*tau*x-sig**2)/2/tau**2)*(1-erf( (-tau*x+sig**2)/sqrt2/tau/sig))

def step_function(x,x0,amp,sig,c):
    x = -(x-x0) # time is inverted
    return amp*0.5*(1-erf(-x/sqrt2/sig))+c

def step_tau(x,x0,amp,sig,tau,c):
    x = -(x-x0) # time is inverted
    return amp*(conv_gauss_and_exp(x,sig,tau)-conv_gauss_and_const(x,sig))+c

def s2mm(s): return s*3e8/2*1e3
def ps2mm(s): return s2mm(s/1e12)
def mm2s(pos): return 2*pos*1e-3/3e8
def mm2ps(pos): return mm2s(pos)*1e12
def mm2fs(pos): return mm2s(pos)*1e15

def _modtime(fname): return os.stat(fname).st_mtime

def dofit(x,y,fname="",scan_motor="",detector="",fit_with_tau=True):

    #model = lmfit.Model(step_function)
    if fit_with_tau:
        model = lmfit.Model(step_tau)
        pars = model.make_params(
           c = y[:3].mean(),
           amp = y[-3].mean()-y[:3].mean(),
           x0 = x.mean(),
           sig = (x.max()-x.min())/20,
           tau = (x.max()-x.min())/10
        )
    else:
        model = lmfit.Model(step_function)
        pars = model.make_params(
           c = y[:3].mean(),
           amp = y[-3].mean()-y[:3].mean(),
           x0 = x.mean(),
           sig = (x.max()-x.min())/20,
        )
    print("###PARS###")
    print(pars)
    print("###PARS###")
        
    res=model.fit(y,x=x,params=pars)
    res.plot()
    ax = plt.gcf().axes[1]
    pos = res.best_values["x0"]
    fwhm =res.best_values["sig"]*2.35
    is_time = abs(res.best_values["sig"]) < 1e-6
    if is_time:
        info_str = "center @ %.3e s\n"%pos
        info_str += "fwhm  %.3e s"%fwhm
        if 'tau' in res.best_values:
            tau = res.best_values["tau"]
            info_str += "\ntau %.3e s"%tau
    else:
        info_str = "center @ %.3f mm (%.3f ps)\n"%(pos,mm2ps(pos))
        info_str += "fwhm  %.3f mm (%.1f fs)"%(fwhm,mm2fs(fwhm))
        if 'tau' in res.best_values:
            tau = res.best_values["tau"]
            info_str += "\ntau %.3f mm (%.1f fs)"%(tau,mm2fs(tau))
    print(info_str)
    ax.text(0.5, 0.2, info_str, transform=ax.transAxes)
    plt.title(pathlib.Path(fname).stem)
    plt.ylabel(detector)
    plt.xlabel(scan_motor)
    for ax in plt.gcf().axes: ax.grid()
    #print(res.fit_report())


def ana_run(fname="YAGSample_FEL0.3_2705_100mbar_026_scan_info.json",detector="SLAAR11-LSCP1-FNS:CH0:VAL_GET",fit=True,as_time=True,fit_with_tau=False):
    info = read_json(fname)
    scan = []
    data = []
    alldata = []
    allpos = []
    is_time_scan = np.max(np.abs(info["scan_readbacks"])) < 1e-9
    if not is_time_scan and as_time:
        info["scan_readbacks"] = mm2s(np.asarray(info["scan_readbacks"]))
        info["scan_readbacks"] -= np.mean(info["scan_readbacks"])
        is_time_scan = True
    for pos,_fname in zip(info["scan_readbacks"],info["scan_files"]):
        if pathlib.Path(fname).exists():
            try:
                temp = ana_h5(_fname,detector=detector)
                data.append( temp["mean"] )
                scan.append(pos)
                if not is_time_scan:
                    _pos = mm2s(pos)+temp["etof"]
                else:
                    _pos = pos + temp["etof"]
                allpos.append(_pos)
                alldata.append( temp["diode"] )
            except OSError:
                pass
    scan = np.asarray(scan)
    data = np.asarray(data)

    allpos = np.hstack(allpos)
    allpos -= np.mean(allpos)
    alldata = np.hstack(alldata)
    xrebin = np.arange(*np.percentile(allpos,(5,95)),20e-15)
    idx = np.digitize(allpos,xrebin)
    yrebin = np.bincount(idx,alldata)[:len(xrebin)]/np.bincount(idx)[:len(xrebin)]
    if fit:
        dofit(scan,data,scan_motor=info["scan_motor"],detector=detector,
             fname=fname,fit_with_tau=fit_with_tau)
        dofit(xrebin,yrebin,scan_motor=info["scan_motor"],detector=detector,
             fname=fname,fit_with_tau=fit_with_tau)
    return dict(x=scan,y=data,xrebin=allpos,yrebin=alldata)

def ana_select(n=None,folder=json_folder,as_time=False,fit_with_tau=False):
    if isinstance(folder,str): folder = pathlib.Path(folder)
    files = list(folder.glob("*"))

    if len(files) == 0:
        print("No files in folder",str(folder))
        return 

    files = sorted(files,key=_modtime)

    if n is not None:
        fname = files[int(n)]
    else:
        for numfile,fname in enumerate(files):
            print(numfile,fname.stem)
            ans = ""
        while not isinstance(ans,int):
            ans = input("Choose your file number ")
            if ans == "":
                ans= -1
            else:
                try:
                    ans = int(ans)
                except ValueError:
                    pass
        fname = files[ans]
    return ana_run(fname,as_time=as_time,fit_with_tau=fit_with_tau)
