import datastorage
import numpy as np
import matplotlib.pyplot as plt
from swissfel_reader import HDF5_swissfel


# PBPS : photonics beam position scattering

# pips raw: SAROP21-CVME-PBPS2:Lnk9Ch7-DATA
# pips calibrated: SAROP21-CVME-PBPS2:Lnk9Ch7-DATA-CALIBRATED

# PBPS133

def show_correlation(data):
    ipm = data.monatt
    pips = data.pips
    diodes = ["u","d","l","r"]
    idx = ipm.u.integral > 1
    fig,ax = plt.subplots( 4,2,sharex=True,sharey=True)
    ax = ax.ravel()
    n = 0
    for i in range(4):
      for j in range(i+1,4):
          d1 = diodes[i]
          d2 = diodes[j]
          r = ipm[d1].sum[idx] / ipm[d2].sum[idx]
          r = r/np.median(r)
          noise = r.std()
          ax[n].plot(r)
          ax[n].set_title("normalized %s.sum/%s.sum, noise %.1f%%"%(d1,d2,noise*100))
          n += 1
    s = np.asarray( [ipm[diode].sum[idx] for diode in "udlr" ] )
    s = s.sum(axis=0)
    r = pips.integral[idx]/s
    r = r/np.median(r)
    noise = r.std()
    ax[n].plot(r)
    ax[n].set_title("normalized pips/sum(ipm), noise %.1f%%"%(noise*100))
    plt.figure()
    plt.hist(r,np.arange(0.8,1.2,0.02))


def readPBPS(fname):
    data = HDF5_swissfel(fname)
     
    d = datastorage.DataStorage()
    base = data.SAROP21_CVME_PBPS2
    up   =  base.Lnk9Ch12.DATA
    down =  base.Lnk9Ch13.DATA
    right = base.Lnk9Ch14.DATA
    left  = base.Lnk9Ch15.DATA
    pips  = base.Lnk9Ch7.DATA
    pulseid = base.Lnk9Ch7.DATA.pulse_id
    d.monatt = datastorage.DataStorage( u = up , d = down, l = left, r = right  )
    d.pips   = pips
    d.data   = data
    d.pulseid = pulseid
    return d


    
    
