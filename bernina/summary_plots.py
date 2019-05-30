import tiox_jf_utils
import numpy as np
import datastorage
import h5py
import trx
import matplotlib.pyplot as plt

cmap = plt.cm.jet

def color(i,N): return cmap(i/N)

def plot_lambda_corr():
    folders = ["../../tiox/l03/run02/","../../tiox/l03/run03/"]
    norm = (0.7,0.9)
    data_nocorr = tiox_jf_utils.analyzeRuns(folders,plot='none',rescale_peak=False,norm=norm)
    data_corr = tiox_jf_utils.analyzeRuns(folders,plot='none',rescale_peak=True,norm=norm)
    fig,ax=plt.subplots(2,2,sharex=True,sharey='col')
    N = len(data_corr.offs)
    q = data_corr.q

    for i,y in enumerate(data_nocorr.offs):
        ax[0][0].plot(q,y,color=color(i,N))
    ax[0][0].set_title("off, no correction")

    for i,y in enumerate(data_nocorr.diffs):
        ax[0][1].plot(q,y,color=color(i,N))
    ax[0][1].set_title("diffs, no correction")


    for i,y in enumerate(data_corr.offs):
        ax[1][0].plot(q,y,color=color(i,N))
    ax[1][0].set_title("off, with correction")


    for i,y in enumerate(data_corr.diffs):
        ax[1][1].plot(q,y,color=color(i,N))
    ax[1][1].set_title("diffs, with correction")


