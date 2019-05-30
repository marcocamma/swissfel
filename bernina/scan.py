import os
import read_diodes
from matplotlib import pyplot as plt
from functools import lru_cache
import joblib
import collections
import numpy as np
import datastorage
from lmfit.models import StepModel
step = StepModel()
import bernina
plt.ion()

#analyzed = {"pos": [], "scan": [], "scan_no": [], "scan_label": []}
#analyzed = {}

#def clear_analyzed():
#    global analyzed
#    parse = ["pos","scan","scan_no","scan_label"]
#    for arg in ["scan_no","scan_label"]:
#        analyzed[arg] = np.zeros(0)
#    for arg in ["pos","scan"]:
#        analyzed[arg] = np.zeros(0)
#    #analyzed = {"pos": [], "scan": [], "scan_no": [], "scan_label": []}

# clear_analyzed()

#def sort_analyzed():
#    global analyzed
#    idx = np.argsort(analyzed["scan_no"])
#    for item in analyzed:
#       analyzed[item] = analyzed[item][idx]

cache_folder = os.path.dirname(__file__) + "/cache"
cache = joblib.Memory( cache_folder, verbose=False )


g_folder = "/sf/bernina/config/com/data/scan_info"

scan_cache = joblib.Memory( "./" , verbose=False )

def scan_motor(scan_info):
    label = scan_info.scan_parameters['Id'][0].split(':')[0]
    return bernina.aliases[label]["alias"]   

#@lru_cache(maxsize=None)
@cache.cache
def anaFile(fname):
    print("Ana",fname)
    data = read_diodes.readPBPS(fname)
    ipm = data.monatt
    mon  = ipm.u.sum + ipm.d.sum + ipm.l.sum + ipm.r.sum
    dio  = data.pips.sum
    idx = mon>-50
    if idx.sum() < 4:
        return np.nan,mon,dio
    else:
        return -dio[idx].sum()/mon[idx].sum(),mon,dio

def readJson(n=1):
    import json
    import glob
    files = glob.glob(g_folder + "/scan%03d*"%n)
    fname = files[0]
    with open(fname) as f:
      a = json.load(f)
    return datastorage.DataStorage(a)
 
def anaScan(n=1,plot=True):
    scan_info = readJson(n=n)
    files = np.asarray(scan_info.scan_files)[:,0]
    scanpos = np.asarray(scan_info.scan_values)[:,0]
    s = []
    spos = []
    for ifile,fname in enumerate(files):
        try:
            s.append( anaFile(fname)[0] )
            spos.append( scanpos[ifile] )
        except (ValueError,OSError):
            print("Cound not analyze file",fname)
    spos = np.asarray(spos)
    s    = np.asarray(s)
    if plot:
        plt.plot(spos,s)
        plt.xlabel("Scan Pos")
        plt.ylabel("PIPS current / PBPS current")
        plt.title("scan%03d"%n)
        plt.tight_layout()
    return spos,s

@scan_cache.cache
def _anaScan(n,n_label=[]):
    pos, scan = anaScan(n,plot=False)
    return dict(scan_no=n,scan_label=n_label,pos=pos,scan=scan) 

def anaScanSet(scan_no=[],scan_label=None,plot=False):
    for n,n_label in zip(scan_no,scan_label):
        ret = _anaScan(n,n_label)
    if plot:
        plt.figure()
        n_sorted, n_label_sorted = zip(*sorted(zip(scan_no,scan_label)))
        for n,n_label in zip(n_sorted,n_label_sorted):
            ret = _anaScan(n,n_label)
            plt.plot(ret["pos"],ret["scan"],label="scan %d: %s" %(n,ret["scan_label"]))
        plt.ylabel("Normalized Intensity")
        plt.grid()
        plt.legend()
        plt.tight_layout()

#def getAnalyzed(scan):
#    global analyzed
#    for i,sn in enumerate(list(analyzed["scan_no"])):
#        if sn == scan:
#            p = list(analyzed["pos"][i])
#            s = list(analyzed["scan"][i])
#    return p,s

#def anaScanSet(scan_no=[],scan_label=None,plot=False,clear=False):
#    global analyzed
#    if clear: clear_analyzed()
#    p = []; s = []
#    if not np.size(analyzed["scan_no"]):
#        for scan in scan_no:
#            print("ANALYZING SCAN #%d..." % scan)
#            pt, st = anaScan(scan,plot=False)
#            p.append(pt)
#            s.append(st)
#        analyzed["pos"] = np.asarray(p)
#        analyzed["scan"] = np.asarray(s)
#        analyzed["scan_no"] = np.asarray(scan_no)
#        analyzed["scan_label"] = np.asarray(scan_label)
#    else:     
#        for scan in set(scan_no).intersection(set(list(analyzed["scan_no"]))):
#            print("PREVIOUSLY ANALYZED SCAN #%d..." % scan)
#            pt, st = getAnalyzed(scan)
#            p.append(pt)
#            s.append(st)
#        for i,scan in enumerate(set(scan_no).difference(set(list(analyzed["scan_no"])))):
#            print("ANALYZING NEW SCAN #%d..." % scan)
#            pt, st = anaScan(scan,plot=False)
#            p.append(pt)
#            s.append(st)
#            pa = analyzed["pos"] 
#            analyzed["pos"] = np.reshape(np.append(pa,pt),(np.shape(pa)[0]+1,np.shape(pa)[1]))
#            sa = analyzed["scan"] 
#            analyzed["scan"] = np.reshape(np.append(sa,st),(np.shape(sa)[0]+1,np.shape(sa)[1]))
#            analyzed["scan_no"] = np.append(analyzed["scan_no"],scan)
#            analyzed["scan_label"] = np.append(analyzed["scan_label"],scan_label[i])
#    sort_analyzed()
#    if plot:
#        plt.figure()
#        if scan_label is not None: 
#            for i in range(len(scan_no)):
#                plt.plot(p[i],s[i],label="scan%d: %s" % (scan_no[i],scan_label[i]))
#        else:
#            for i in range(len(scan_no)):
#                plt.plot(p[i],s[i])
#        plt.ylabel("Normalized Intensity")
#        plt.grid()
#        plt.legend()

def fit_scan(t,s):
    idx = np.isfinite(s) & np.isfinite(t) & (np.abs(t) < 2e-12)
    if idx.sum() < 10: return np.nan
    t = t[idx]
    s = s[idx]
    pars = step.make_params(amplitude = -0.02, center=0.2e-12,sigma=0.2e-12 )
    res = step.fit(s,x=t,params=pars)
    print(res,res.best_values)
    return res.best_values["center"]

plt.rcParams['font.size'] = 14

def anaStability():
    scans = range(8,37)
#    scans = range(8,12)
    fig,ax=plt.subplots(1,2,sharey=True,gridspec_kw = {'width_ratios':[1, 2]} )
    for scan in scans:
        try:
            t,s=anaScan(scan,plot=False)
            idx = np.abs(t) < 2e-12
            t = t[idx]; s = s[idx]
            ax[0].plot(s,1e12*t,label="%s"%scan)
            fitpos = fit_scan(t,s)
            print("DDADSADA",scan,fitpos)
            ax[1].plot( scan,fitpos*1e12,"o" )
        except ValueError:
            print("could not read scan",scan)
    ax[0].legend()
    ax[0].set_xlabel("Bi 111")
    ax[0].set_ylabel("time (ps)")
    ax[1].set_xlabel("scan num")

