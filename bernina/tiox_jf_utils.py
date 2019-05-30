import numpy as np
import pathlib
import glob
import os
import datastorage
import h5py
import sys
sys.path.append("/afs/psi.ch/sys/psi.ra/Programming/psi-python36/5.0.1/lib/python3.6/site-packages/")
try:
    import jungfrau_utils as ju
except ImportError:
    print("Can't import jungfrau_utils")
    pass
import trx
import re
import matplotlib.pyplot as plt
import collections
import joblib
import lmfit
import time
from collections import OrderedDict
#cache_folder = os.path.dirname(__file__) + "/cache_azav"
#cache = joblib.Memory( cache_folder )

g_folder = "/sf/bernina/config/com/data/scan_info"
g_folder = "../../scan_info"

TSHIFT = 37.5e-12



plt.ion()

_timeInStr_regex = re.compile(r'_(-?\d+\.?\d*(?:fs|ps|ns|us|ms|s))', re.UNICODE)

def wp_to_E(angle):
    return 2.09e-2+2.029*np.sin( 2*np.deg2rad(angle) + 0.7572)**2

    

def getDelayFromString(string) :
    match = _timeInStr_regex.search(string)
    return match and match.group(1) or None



azav_pars_default = dict( dist=0.072, pixel = 7.5e-05, xcen = 376.1, ycen  = 202.1, wavelength = 1.8587e-10 )
azav_pars_default = dict( dist=0.072, pixel = 7.5e-05, xcen = 372.1, ycen  = 202.1, wavelength = 1.8587e-10 )


# tuesday
azav_pars_default = dict( dist=0.072, pixel = 7.5e-05, xcen = 371.26, 
                    ycen  = 149.04, rot1=0.00601,rot2=0.027845,wavelength = 1.8587e-10 )

# wed
azav_pars_default = dict( dist=0.063, pixel = 7.5e-05, xcen = 389.46, 
                    ycen  = -32.935, rot1=-0.00159,rot2=0.01154,wavelength = 1.8587e-10 )

# based on run 161 (22.12.17)
azav_pars_default = dict( dist=0.06097, pixel = 7.5e-05, xcen = 406.728, ycen  = -33.145,rot1=0.015,rot2=0.01118,wavelength = 1.899e-10 )


def find_last_pedestal(folder="/sf/bernina/config/com/data/JF_pedestal/"):
  files=list(glob.glob( os.path.join(folder,"pedestal_*res.h5")) )
  files.sort()
  return files[-1]

def expandModule(img):
    assert img.shape == (512,1024)
    nimage = np.zeros( (514,1032),dtype=img.dtype )
    for col in range(4):

      #print(col,"orig",1+256*col,256*(col+1)-1)
      #print(col,"new",2+258*col,258*(col+1)-2)

      col_orig = slice( 1+256*col, 256*(col+1)-1  )
      col_new  = slice( 2+258*col, 258*(col+1)-2  )

      nimage[ 0:255,col_new] = img[0:255,col_orig]

      nimage[259:514,col_new] = img[257:512,col_orig]

    return nimage



def expandImage(img):
  nrow = (512+2)*3+2*36
  ncol = 1032
  nimg = np.zeros( (nrow,ncol),dtype=img.dtype )
  for n in range(3):
    row_orig = slice( n*512, (n+1)*512 )
    row_new  = slice( n*514+n*36, (n+1)*514+n*36 )
    nimg[row_new] = expandModule( img[row_orig] )
  return nimg

def mask_gap(img):
  return img == 0

def readF(fname,what):
    if not isinstance(what,(list,tuple)): what = (what,)
    f = h5py.File(fname,"r")
    ret = [f[w][:] for w in what]
    if len(ret) == 1: ret=ret[0]
    f.close()
    return ret

#def getFluo(img):
#    idx_fluo = (img<5.3) & (img>3.8)
#    idx_2fluo = (img>7.7) & (img<10.1)
#    idx_1fluo_1elastic = (img>10.1) & (img<12.2)
#    return p1+p2+p3
 
def filterImage(img):
    idx_fluo = (img<5.7)
    idx_2fluo = (img>7.7) & (img<=10.1)

    # remove high intensity
    idx_high_intensity = img > 6.6*50

    idx_1fluo_1elastic = (img>10.1) & (img<12.2)
    img[idx_1fluo_1elastic] -= 4.5
#    img[idx_1fluo_1elastic]  = 0.

    img[    idx_2fluo ] = 0
    img[ idx_high_intensity ] = 0
    img[idx_fluo] = 0

#    # 5th harmonic
#    img[ idx_1fluo_1elastic ] = 0; # it is around 10 ...
#    E = 3.3*5
#    idx_5th_2 = (img>19) & (img<21)
#    img[idx_5th_2] = 0

    return img#,fluo


class JFImages(object):

    def __init__(self,fname=None,config_folder='auto',
            azav_pars=azav_pars_default,mask=None,gainfile='auto',
            pedestal='auto',N=None):
        if config_folder == 'auto':
            thisfile = pathlib.Path(__file__)
            config_folder = str(thisfile.parents[2])

        self.conf_folder = config_folder
        
        # find gainfile and read gains
        if gainfile == 'auto':
            gainfile = os.path.join(config_folder,"gains.h5")
        self.gainfile = gainfile
        self.gains = readF(gainfile,"gains")

        # find pedestal if needed
        if pedestal == 'auto' : pedestal = find_last_pedestal( os.path.join(config_folder,"JF_pedestal") )
        self.pedestalfile = pedestal
        self.pedestals,bad_pixels = readF(pedestal,["gains","pixelMask"])

        mask_total = expandImage( bad_pixels > 0 ) ## it was == 0
        if mask is not None: mask_total = mask_total | mask
        # add gaps in mask
        gap_mask = expandImage( np.ones( (3*512,1024) ) ) == 0
        mask_total = mask_total | gap_mask
        self.bad_pixel_mask = mask_total #,"y<250")
        self.total_mask = self.bad_pixel_mask

        # prepare pyFAI integrator
        self.azav_pars = azav_pars
        self.ai = trx.azav.getAI(azav_pars) if azav_pars is not None else None

        self.N=N

        if fname is not None: self.readJF(fname)

    def readJF(self,fname):
          self.fname = fname
          self.basename = os.path.basename(fname)
          f = h5py.File(fname,"r")
          pulseid = f["jungfrau/pulse_id"][:]
          withBeam = np.squeeze( np.argwhere( pulseid%10 == 0 ) )
          withBeamAndLaser   = np.squeeze( np.argwhere( pulseid % 20 == 0 ) )
          withBeamAndNoLaser = np.squeeze( np.argwhere( pulseid % 20 == 10 ) )
          self.pulseid = pulseid
          if self.N is not None:
              withBeam = withBeam[:self.N]
              self.pulseid = self.pulseid[:self.N]
              withBeamAndLaser = withBeamAndLaser[:int(self.N/2)]
              withBeamAndNoLaser = withBeamAndNoLaser[:int(self.N/2)]
          self.idx=withBeam
          self.withBeamAndLaser = withBeamAndLaser
          self.withBeamAndNoLaser = withBeamAndNoLaser
          self.imgs = f["jungfrau/data"]

    def _correct(self,img,laser=None,energy_filter=True,expand=True):
        
        print("in _correct","%20s"%self.basename,end = " ")

        if isinstance(img,int):
            print("%7d"%img,end=" ")
            if laser is None:
                img_no = self.idx[img]
            elif laser == 'on':
                img_no = self.withBeamAndLaser[img]
            elif laser == 'off':
                img_no = self.withBeamAndNoLaser[img]
            print("laser",laser,"idx: %7d"%img_no,end="\r")
            img = self.imgs[img_no]
        else:
            print("",end="\r")
        img = ju.apply_gain_pede(img,G=self.gains,P=self.pedestals)
        
        if energy_filter: img = filterImage(img)
        if expand: img = expandImage(img)

        return img

    def correct(self,imgs,laser=None,energy_filter=True,expand=True):
        if isinstance(imgs,int) or (isinstance(imgs,np.ndarray) and imgs.ndim == 2):
            return self._correct(imgs,laser=laser,energy_filter=energy_filter,
                    expand=expand)
        else:
            if imgs=="all":
                if laser == "on" :
                    imgs = range(len(self.withBeamAndLaser))
                elif laser == "off":
                    imgs = range(len(self.withBeamAndNoLaser))
                else:
                    imgs = range(len(self.idx))
            imgs = [self._correct(img,laser=laser,energy_filter=energy_filter,
                expand=expand) for img in imgs]
            imgs = np.asarray(imgs)
        return imgs

    def _azav(self,img,correct=True,npt_radial=1600,return_err=False,laser=None,energy_filter=True,expand=True):
        # correct image if needed
        if correct: 
            img = self.correct(img,laser=laser,energy_filter=energy_filter,expand=expand)
        else:
            if expand: img = expandImage(img)

        if self.ai is not None:
            self.q,i,err = trx.azav.do1d(self.ai,img,mask=self.total_mask,\
                       npt_radial=npt_radial,dark=0)
            if return_err:
                return i,err
            else:
                return i
        else:
            return None

    def azav(self,imgs,correct=True,npt_radial=1600,laser=None,force=False,timing=False,energy_filter=True,save=True,expand=True):
        if isinstance(imgs,int) or (isinstance(imgs,np.ndarray) and imgs.ndim == 2):
            return self._azav(imgs,correct=correct,npt_radial=npt_radial,laser=laser,energy_filter=energy_filter,expand=expand)
        else:
            #print("\n\n|||||",imgs)
            if imgs=="all":
                if laser == "on" :
                    imgs = range(len(self.withBeamAndLaser))
                    azav_fname = self.fname[:-3] + "_az_on.h5"
                elif laser == "off":
                    imgs = range(len(self.withBeamAndNoLaser))
                    azav_fname = self.fname[:-3] + "_az_off.h5"
                else:
                    azav_fname = self.fname[:-3] + "_az.h5"
                    imgs = range(len(self.idx))
            else:
                    azav_fname = self.fname[:-3] + "_az.h5"
            #print("\n\n",save)
            if isinstance(save,str): azav_fname = save
            if os.path.isfile(azav_fname) and not force:
                temp = datastorage.read(azav_fname)
                self.q = temp["q"]; azav = temp["azav"]
                return azav
            else:
                t0 = time.time()
                azav = np.asarray( [self._azav(imgs[i],correct=correct,
                    npt_radial=npt_radial,laser=laser,
                    energy_filter=energy_filter,expand=expand) for i in imgs] )
#                azav=joblib.Parallel(n_jobs=4, backend="multiprocessing")( (joblib.delayed(expose)(self,imgs[i],correct=correct,npt_radial=npt_radial,laser=laser) for i in imgs)  )
#                azav = joblib.
                if timing:
                    dt = time.time()-t0
                    print("Time needed for %d images: %.1fs (%.1f Hz)"%(len(imgs),dt,len(imgs)/dt))
                if laser is None:
                    pulseid =  self.pulseid[self.idx]
                elif laser == "on":
                    pulseid =  self.pulseid[self.withBeamAndLaser]
                elif laser == "off":
                    pulseid = self.pulseid[self.withBeamAndNoLaser]
                #print('saving',azav_fname)
                if save:
                    #print('saving',azav_fname)
                    temp = datastorage.DataStorage( q=self.q, azav=azav,
                            pulse_id=pulseid )
                    temp.save(azav_fname)
            return azav

    def maskFromSpikes(self,imgs=range(100),threshold=2):
        if isinstance(imgs,range): imgs = range(imgs.start,
                min(imgs.stop,len(self.withBeamAndNoLaser)))
        imgs = self.correct(imgs,laser='off')
        img = imgs.mean(0)
        # send copy to dezinger since it modify in place
        img_dez = trx.azav.dodezinger(self.ai,img.copy(),\
                mask=self.bad_pixel_mask,npt_radial=1600,dezinger=50)
        diff = (img-img_dez)/(img_dez+0.5)
        mask = diff > threshold; # do not use abs here !
        self.mask_spikes = mask
        self.total_mask = (self.bad_pixel_mask | mask )
        return diff,mask

def azwrapper(jfobj,img,correct=True,npt_radial=1600,laser=None):
    return jfobj._azav(img,correct=correct,npt_radial=npt_radial,laser=laser)


jf = JFImages()

def azavRun(fileListOrfolder,force=False,peakThreshold=None):
    if isinstance(fileListOrfolder,(list,tuple,np.ndarray)):
        files = fileListOrfolder
    else:
        files = glob.glob( os.path.join(fileListOrfolder,"*.h5") )
        files = [file for file in files if file.find("_az") == -1]
    for f in files:
        try:
          jf.readJF(f)
          if peakThreshold is not None:
              jf.maskFromSpikes(range(100),threshold=peakThreshold)
          temp = jf.azav("all",laser="off",force=force)
          temp = jf.azav("all",laser="on",force=force)
        except Exception as e:
          print("Failed",e)
          pass

def find_peak(x,i,N=5):
    idx = np.argmax(i)
    poly = np.polyfit( x[idx-N:idx+N],i[idx-N:idx+N],2)
    polyder = np.polyder(poly)
    xmax    = np.roots(polyder)
    return float(xmax)

def Q(x,d,lam):
    return 4*np.pi/lam*np.sin( np.arctan(x/d) / 2 )

Qfit = lmfit.Model(Q)

def refine_pos_lambda(q,offs,ons,fix_lam=False,fix_dist=False):
    lam_fit = []
    d_fit = []
    r = azav_pars_default['dist']*np.tan(q*azav_pars_default['wavelength']*1e10/2/np.pi)

    idx1 = (q>1.4) & (q<1.5)
    idx2 = (q>1.5) & (q<1.95)
    idx3 = (q>2.94) & (q<3.1)
    # In [15]: 4*np.pi/azav_pars_default['wavelength']/1e10*np.sin(np.deg2rad([25.6028
    #    ...: ,31.1044,35.14268,53.7247])/2)
    #Out[15]: array([ 1.4662243 ,  1.77422193,  1.99773567,  2.99004578])
    p1 = 1.4662 # 201 beta
    p2 = 1.7742 # 110 beta
    p3 = 2.9900 # 204 beta
    yf = (p1,p2,p3)

    for i,(off,on) in enumerate(zip(offs,ons)):
        try: 
            r1 = find_peak(r[idx1],off[idx1])
            r2 = find_peak(r[idx2],off[idx2])
            r3 = find_peak(r[idx3],off[idx3])
            xf = (r1,r2,r3)
            pars = Qfit.make_params( lam = 1.878, d = 0.063 )
            if fix_lam:  pars['lam'].set(vary=False)
            if fix_dist: pars['d'].set(vary=False)
         
            # fit
            resfit = Qfit.fit(yf,x=xf,params=pars)
            lam_shot = resfit.best_values['lam']
            d_shot = resfit.best_values['d']
            # save fit parameters
            lam_fit.append( lam_shot )
            d_fit.append( d_shot )
         
            qshot = Q(r,d_shot,lam_shot)
         
            offs[i] = np.interp(q,qshot,off)
            ons[i]  = np.interp(q,qshot,on)
        except:
            d_fit.append( np.nan )
            lam_fit.append( np.nan )

    fitpars = datastorage.DataStorage( lam = np.asarray(lam_fit),dist=np.asarray(d_fit) )
    return offs,ons,fitpars

def refine_lambda(q,offs,ons):
    lam_fit = []
    reference = 1.775

    for i,(off,on) in enumerate(zip(offs,ons)):
    
        qmax_pos = find_peak(q,off)
        scaling  = reference/qmax_pos
        offs[i]  = np.interp(q,q*scaling,off)
        ons[i]   = np.interp(q,q*scaling,on)

        lam_fit.append( 1.879*scaling )


    fitpars = datastorage.DataStorage( lam = np.asarray(lam_fit) )
    return offs,ons,fitpars

def _analyzeRun(fileListOrfolder,delays=None,doAzAv=False,norm=(0.7,0.9),refine_pars=2,force=False,peakThreshold=None):
    """ 
       for data saved on wed. night (tiox/l03/run*) 
       delays is needed is you send a file list
    """
    if doAzAv: azavRun(fileListOrfolder,force=force,peakThreshold=peakThreshold)

    
    # find files if needed
    if isinstance(fileListOrfolder,(list,tuple,np.ndarray)):
        files = fileListOrfolder
        # take only data for which on and off exists
        files = [f for f in files if os.path.isfile(f.replace(".h5","_az_off.h5"))]
        files = [f for f in files if os.path.isfile(f.replace(".h5","_az_on.h5"))]
        files = [f.replace(".h5","_az_off.h5") for f in files]
    else:
        files = glob.glob(os.path.join(fileListOrfolder,"*_az*"))

        # take only data for which on and off exists
        files = [ f for f in files if f.replace("off","on") in files ]

        # the macro will then take care of using the on
        files = [ f for f in files if f.find("off") > 0 ]

        # exctract delay from filename
        delays = [trx.utils.strToTime(getDelayFromString(fname)) for fname in files]
        delays = np.asarray(delays)


    if len(files) == 0:
        print("No files found in",fileListOrfolder)
        return

    #sort based on delay
    if len(files) != len(delays):
        n = min(len(files),len(delays))
        files = files[:n]
        delays= delays[:n]
    
    idx = np.argsort(delays)
    files = np.asarray(files)
    files = files[idx]
    delays = delays[idx]

    # read data
    offs = np.asarray( [ datastorage.read(f).azav.mean(axis=0) for f in files ] )
    on_files = [f.replace("off","on") for f in files]
    ons  = np.asarray( [ datastorage.read(f).azav.mean(axis=0) for f in on_files ] )
    q = datastorage.read(files[0]).q

    # normalize if needed
    if norm is not None:
        idx = (q>=norm[0]) & (q<=norm[1])
        offs_norm = offs[:,idx].mean(1)
        offs      = offs/offs_norm[:,np.newaxis]
        ons_norm  = ons[:,idx].mean(1)
        ons       = ons/ons_norm[:,np.newaxis]


    if refine_pars == 2:
        offs,ons,pars = refine_pos_lambda(q,offs,ons,fix_lam=False,fix_dist=False)
    elif refine_pars == 1:
        offs,ons,pars = refine_pos_lambda(q,offs,ons,fix_dist=True,fix_lam=False)
    elif refine_pars == 3:
        offs,ons,pars = refine_pos_lambda(q,offs,ons,fix_dist=False,fix_lam=True)
    else:
        pars = None
    

    diffs = ons-offs

    return datastorage.DataStorage(q=q,diffs=diffs,ons = ons, offs = offs, delays=delays, corrections=pars)


def analyzeRuns(folders,delays=None,doAzAv=True, plot = "diffs",norm=(0.7,0.9),refine_pars=2,title=None,force=False,exclude=None,peakThreshold=None):
    """ for data saved on wed. night """
    if force: doAzAv = True
    if title is None: title = str(folders)
    if not isinstance(folders,(list,tuple)): folders = (folders,)
    if delays is None: delays = (1,)*len(folders)
    results = [_analyzeRun(folder,delays=delay,norm=norm,refine_pars=refine_pars,doAzAv=doAzAv,force=force,peakThreshold=peakThreshold) for (delay,folder) in zip(delays,folders)]

    q = results[0].q
    # stack things together
    delays = np.hstack( [r.delays for r in results] )
    diffs = np.vstack( [r.diffs for r in results ] )
    ons = np.vstack( [r.ons for r in results ] )
    offs = np.vstack( [r.offs for r in results ] )
    if results[0]['corrections'] is not None:
        corrections = datastorage.DataStorage()
        for k in results[0]['corrections'].keys():
            corrections[k]=np.hstack( (r.corrections[k] for r in results) )
    else:
        corrections = None

    if delays.min()>200: delays = wp_to_E(delays)

    # resort
    idx = np.argsort(delays)
    delays = delays[idx]
    diffs  = diffs[idx]
    ons  = ons[idx]
    offs  = offs[idx]

    if exclude is not None:
       offs = np.delete(offs,exclude,axis=0)
       ons = np.delete(ons,exclude,axis=0)
       diffs = np.delete(diffs,exclude,axis=0)
       delays = np.delete(delays,exclude,axis=0)



    #trx.utils.plotdiffs( dict( q = q,diffs=diffs,scan=delays) )
    if not ('none' in plot):
        fig = plt.figure(title)
        fig.clear()
    if not isinstance(plot,(list,tuple)): plot = (plot,)
    if 'diffs' in plot: trx.utils.plotdiffs( dict( q = q,diffs=diffs,scan=delays),fig=fig )
    if 'ons' in plot: trx.utils.plotdiffs( dict( q = q,diffs=ons,scan=delays),fig=fig )
    if 'offs'  in plot: trx.utils.plotdiffs( dict( q = q,diffs=offs,scan=delays),fig=fig )
    if not ('none' in plot):
        plt.title(title)
        plt.legend( list(map(trx.utils.timeToStr,delays)) ,ncol=6 )
    return datastorage.DataStorage(q=q,diffs=diffs,ons = ons, offs = offs ,delays=delays,scan=delays,folders=folders,title=title,corrections=corrections)

def changeFolder(fullname,newfolder):
    return os.path.join(newfolder,os.path.basename(fullname))

def readJson(n=1,new_folder=None):
    """ for eco scan files """
    import json
    import glob
    files = glob.glob(g_folder + "/scan%03d*"%n)
    fname = files[0]
    with open(fname) as f:
      a = json.load(f)
    if new_folder is not None:
        files = a['scan_files']; # list of list
        newfiles = []
        for scanstepfiles in files:
            temp = [changeFolder(f,new_folder) for f in scanstepfiles]
            newfiles.append( temp )
        a['scan_files'] = newfiles

    return datastorage.DataStorage(a)


def anaScan(scan_no,scan_folder="../../scan_data",doAzAv=False,force=False,
        peakThreshold=1,norm=(0.7,0.9),refine_pars=2,exclude=None,
        toffset=TSHIFT,save=True,plot=['diffs','ons'],clim='auto',
        plotScan=False,save_xy=False):
    """ 
    
    This is to analyze scans taken with eco

    Parameters
    ----------

    scan_no: int or tuple
        scan(s) to analyze

    scan_folder: string
        where data are to be found (if None use full path in json file

    doAzAv: bool
        if azav should be started, ususlaly false because AzAv taken care
        by other process (autoAnalysisScanFolder)

    force: bool
        Force calculation of azimuthal averages even if presence (internally
        also changes doAzAv to True)

    peakThreshold: float
        used to remove intense peaks from big crystallites.
        units are in azimuathally averaged values (of 100 images image), 
        i.e. 1 means that it removes pixels that differ by more than 100%

    norm: tuple
        q-range to normalizes curves to

    refine_pars: int in 0 to 3
        0: no correction
        1: refine lambda to match the three reference peaks
        2: refine lambda and pos to match three reference peaks
        3: refine pos to match three reference peaks

    exclude: None, int or tuple
        remove curves with indeces in tuple

    toffset: float
        shift delays by this amount (default 37.5ps)

    save: bool or str
        save data if True (automatically determined file name)
        or to th given string (relative path from ./results)
        if False, no data is saved

    save_xy: bool
        Save files in a format compatible with refinement programs
        (twotheta,y) for all time delays

    plot: 'none' or list_of_things_to_plot [offs,ons,diffs]
        if not 'none' plot whatever is desidered

    clim: 'auto' or tuple
        clim to use for color coded plots

    plotScan: bool
        if True, make also 2D plots
    """

    if not isinstance(plot,(list,tuple)): plot = (plot,)

    if not isinstance(scan_no,(list,tuple)): scan_no = (scan_no,)

    # read log file
    scan_info = []
    files = []
    delays = []
    for scan in scan_no:
        scan_info = readJson(n=scan,new_folder=scan_folder)
        _files = np.asarray(scan_info.scan_files)[:,-1]
        # means we guessed wrong (just in case)
        if _files[0].find("JF1p5") == -1: _files = np.asarray(scan_info.scan_files)[:,0]
        files.append( _files )
        delays.append( np.asarray(scan_info.scan_values)[:,0]+toffset )

    result = analyzeRuns(files,delays=delays,plot=plot,doAzAv=doAzAv,refine_pars=refine_pars,title=str(scan_no),force=force,norm=norm,exclude=exclude,peakThreshold=peakThreshold)

    q = result.q
    offs  = result.offs
    ons   = result.ons
    diffs = result.diffs
    delays = result.delays
    wav = azav_pars_default['wavelength']*1e10
    result.twotheta = trx.utils.qToTwoTheta(q, asDeg=True,wavelength=wav)
    
    # remove bkg
    qlims = (1,3.5)
    bkg = [(1,1.02), (1.57,1.67), (1.1,1.17),(1.37,1.40),(1.845,1.855),
            (1.93,1.95),(2.05,2.1),(2.74,2.77),(3.06,3.1),(3.4,3.5)]

    qclean,off_sub = trx.utils.removeBackground(q,offs[0],xlims=qlims,
            max_iter=100,background_regions=bkg)

    idx = trx.utils.findSlice(q, qlims )
    bkg = offs[0,idx]-off_sub
    nobkg = datastorage.DataStorage( q=qclean, twotheta=result.twotheta[idx],
            offs = result.offs[:,idx]-bkg, ons  = result.ons[:,idx]-bkg,
            diffs = result.diffs[:,idx], delays = result.delays )
    result.nobkg = nobkg
   
    if save is True or isinstance(save,str):
        if isinstance(save,str):
            outfname = save
        else:
            scan_hash = "_".join( map(str,scan_no) )
            outfname = 'results/results_scan_%s.h5' % scan_hash
        result.save(outfname)

    if save_xy:
        twotheta = result.twotheta
        # create folder
        scan_hash = "_".join( map(str,scan_no) )
        dirname = "scan_" + scan_hash
        os.makedirs(dirname,exist_ok=True)

        for idelay,(delay,on,off) in enumerate(zip(delays,ons,offs)):
            delay_str = trx.utils.timeToStr(delay,fmt="%.1f")
            basename = "scan_%s_%03d_%s_"%(scan_hash,idelay,delay_str)
            for what,what_str in zip( (on,off),("on","off") ):
                fname = basename+what_str+".xy"
                fname = os.path.join(dirname,fname)
                temp = np.vstack((result.twotheta,what))
                np.savetxt(fname,temp.T,fmt="+%.5f")

    if plotScan:
        fig,ax = plt.subplots(2,2,sharey=True)
        ax[0,0].set_title("offs %s"%str(scan_no))
 
        ax[0][0].plot(offs.mean(axis=0),q)
        ax[0][1].pcolormesh(delays,q,offs.T)
 
        ax[1,0].set_title("diffs %s"%str(scan_no))
        ax[1][0].plot(diffs.mean(axis=0),q)
        ax[1][1].pcolormesh(delays,q,diffs.T)

    return result


def integrate_peaks_mruns(runs, peaks = None, plot_idx = False):
   
   if peaks is None:
      peaks = ( (1.23,1.3),(1.44,1.49), (1.715,1.81), (2.17,2.34))
   npeaks = len(peaks)
   fig,ax = plt.subplots(nrows=npeaks,ncols=2, sharex='col',sharey = 'row',gridspec_kw=dict(width_ratios=[1,2]),squeeze=False)
   for a in ax.ravel(): a.clear()
     # ax[0][1].set_title(title)
 
   for run in runs:
      if isinstance(run, int):
         data = anaScan(run, plotScan = False)
         label = 'scan' + str(run)
      else:
         data = analyzeRuns(run, plot = None)
         label = run

      result = integrate_peaks(data,title = 'NoTitle', peaks = peaks, plot = False)
      peak_tr_abs = result.absolute
      peak_tr = result.integral
        
#    f, (ax1, ax2) = plt.subplots(1,2, sharey = 'row')
#    f.subplots_adjust(wspace = 0)
#    ax1.plot(data.delays,data.ons[:,idx].sum(1)/data.ons[:,:].sum(1),'o-')
#    ax2.semilogx(data.delays,data.ons[:,idx].sum(1)/data.ons[:,:].sum(1),'o-')
#    ax1.plot(data.delays,abs(data.diffs[:,idx]).mean(1),'o-')
#    ax2.semilogx(data.delays,abs(data.diffs[:,idx]).mean(1),'o-')
#    ax1.set_xlim([-5e-11,2e-10] )
#    ax2.set_xlim([2e-10,1e-5] )
#    plt.show()
      if plot_idx == True:
         delays = np.arange(0,len(data.delays))*1e-12
      else:
         delays = data.delays
      for ipeak, peak in enumerate(peaks):
          
         ax[ipeak][0].plot(delays*1e12,peak_tr_abs[ipeak],'-o')
         ax[ipeak][0].plot(delays*1e12,peak_tr[ipeak],'-o')
         ax[ipeak][0].set_xlim( (-100,100) )
         ax[ipeak][0].set_title(peak)

         ax[ipeak][1].plot(delays,peak_tr_abs[ipeak],'-o', label = label+'_abs')
         ax[ipeak][1].plot(delays,peak_tr[ipeak],'-o', label = label)
         ax[ipeak][1].set_xscale("log", nonposx='clip')
         ax[ipeak][1].set_xlim( (1e-10,1e-6) )
         ax[-1,1].legend()
         ax[0,1].legend()
      if plot_idx == True:
         ax[ipeak][0].set_xlim((0, 120)) 

def _findFiles(folder):
    files = []
    for runfolder in glob.iglob(folder + "/*"):
        _temp = glob.glob( runfolder + "/*.h5")
        files.extend(_temp)
    return files


def integrate_peaks(diffs,title=None, plot = True, peaks = None):

   # understand if we scan WP or time
   if diffs.scan.max() > 100e-6:
       isWP = True
       plot = False
   else:
       isWP = False

   if peaks is None:
      peaks = OrderedDict()
      peaks[ (1.23,1.3) ] ="lambda/alpha"
      peaks[ (1.44,1.49) ] ="beta"
      peaks[ (1.715,1.84)] = "common"
      peaks[ (2.25,2.297)] ="alpha"
      peaks[ (2.94,3.04)] ="beta"
      peaks[ (3.146,3.203) ] = "alpha"
   else:
      if isinstance(peaks,(list,tuple)):
          temp = dict()
          for p in peaks: temp[p]='peak'
          peaks = temp
   if title is None: 
       if 'title' in diffs:
           title = diffs.title
       else:
           title = diffs.folders

   q = diffs.q
   delays = diffs.delays
   diffs = diffs.diffs

   npeaks = len(peaks)
   peak_tr_abs = []
   peak_tr = []

   if plot == True:   
      fig,ax = plt.subplots(num="peaks-"+title,nrows=npeaks,ncols=2, sharex='col',sharey = 'row',gridspec_kw=dict(width_ratios=[1,2]),squeeze=False)
      for a in ax.ravel(): a.clear()
      ax[0][1].set_title(title)



   if isWP is True and plot:
      fig, ax = plt.subplots(num = "WP_"+title, nrows=npeaks,ncols=1)

      
   for ipeak,(peak,peak_info) in enumerate(peaks.items()):
 
      idx = ( q>=peak[0] ) & (q<=peak[1])
      s  = np.abs(diffs[:,idx]).mean(1)
      s1 = diffs[:,idx].mean(1)
      peak_tr_abs.append(s)
      peak_tr.append(s1)
      if plot == True:
         ax[ipeak][0].set_ylabel(peak_info)
         ax[ipeak][0].plot(delays*1e12,s,'-o')
         ax[ipeak][0].plot(delays*1e12,s1,'-o')
         ax[ipeak][0].set_xlim( (-100,100) )
         ax[ipeak][0].set_title(peak)
 
         ax[ipeak][1].plot(delays,s,'-o')
         ax[ipeak][1].plot(delays,s1,'-o')
         ax[ipeak][1].set_xscale("log", nonposx='clip')
         ax[ipeak][1].set_xlim( (1e-10,1e-6) )
         ax[-1,1].legend( ["abs","sum"] )
      if isWP is True and plot:
         ax[ipeak].set_title(peak)
         ax[ipeak].plot(delays, s, '-o')
         ax[ipeak].plot(delays, s1, '-o')
   return datastorage.DataStorage(absolute=peak_tr_abs, integral=peak_tr,peaks_pos=peaks.keys(),peak_info=peaks.values(),delays=delays,scan=delays)



def plot3d(data):
     from mpl_toolkits.mplot3d import Axes3D

     fig = plt.figure()
     ax = fig.gca(projection='3d')

     ax.plot(x, y, zs=0, zdir='z', label='curve in (x,y)')

def autoAnalysis(folder,force=False,reverse=False):
    while True:
        try:
             files = _findFiles(folder)
             data = [f for f in files if f.find("az") == -1 ]
             if reverse: data = data[::-1]
             for fname in data:
                 if not (fname.replace(".h5","_az_off.h5") in files) or force:
                     try:
                         jf.readJF(fname)
                         temp = jf.azav("all",laser="off",force=force)
                         temp = jf.azav("all",laser="on",force=force)
                     except OSError:
                         pass
        except KeyboardInterrupt:
            break
        print(time.asctime(),"Auto analysis running, ... waiting for files ...")
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            break

_svd_ret = collections.namedtuple("svd_return",["basis","s",\
           "population","data"])

def svd(data,ncomp=None,truncate=True):
  """ do SVD based cleaning 
      
      Parameters
      ----------
      ncomp : int
        number of components to keep
      truncate : bool
        if True, returns only basis and populations of up to ncomp
  """
  # first index is time/T/P
  u,s,v = np.linalg.svd(data,full_matrices=False)
  if ncomp is not None:
    s[ncomp:]=0
    data = np.dot(u,np.dot(np.diag(s),v))
    if truncate:
      v = v[:ncomp]
      u = u[:,:ncomp]
  return _svd_ret(basis=v,s=s,population=u.T,data=data)

def barcode(data=(155,161,162),what='diffs',cmap=plt.cm.bwr,clim=(-0.1,0.2),svdcomp=5):
    if isinstance(data,(int,tuple)): data = anaScan(data,plot=False)
    q = data.q
    y = data[what]
    t = data.delays
    if isinstance(svdcomp,int) and svdcomp>=1: y = svd(y,ncomp=svdcomp).data

    fig,ax=plt.subplots(1,2,sharey=True,figsize=[5,10])
    plt.subplots_adjust(wspace=0,right=0.95,left=0.15,top=0.98)
    
    ax[0].pcolormesh(t*1e12,q,y.T,cmap=cmap,vmin=clim[0],vmax=clim[1])
    ax[0].set_xticks( (0,25,50) )
    ax[0].set_xlim(-5,50)
    ax[0].set_yticks( np.arange(1.5,3.5,0.5) )
    ax[0].set_ylabel(r"scattering vector ($\AA^{-1}$)")
    ax[0].set_xlabel("time (ps)")

    ax[1].pcolormesh(t,q,y.T,cmap=cmap,vmin=clim[0],vmax=clim[1])
    ax[1].semilogx()
    ax[1].set_xlim(50e-12,1e-6)
    ax[1].set_ylim(1.2,3.3)
    ax[1].set_xlabel("time (s)")
    plt.sca(ax[1])
    plt.xticks( (1e-9,1e-8,1e-7,1e-6),("1 ns"," ", " ",r"1$\mu$s") )
    ax[0].grid(ls='--',color="0.9")
    ax[1].grid(ls='--',color="0.9")

def getQ(q,y,Q):
    idx = np.argmin( np.abs(q-Q) )
    return y[:,idx]

def barcode_shorttimes( data=(161,162),what='diffs',cmap=plt.cm.bwr,\
    clim=(-0.1,0.1),svdcomp=10):
    if isinstance(data,(int,tuple)): data = anaScan(data,plot=False)
    q = data.q
    y = data[what]
    t = data.delays

    tm = 100e-12
    idx = t<tm
    t = t[idx]*1e12
    y = y[idx]
    if isinstance(svdcomp,int) and svdcomp>=1: y = svd(y,ncomp=svdcomp).data

    c1 = getQ(q,y,1.453)
    c2 = getQ(q,y,1.468)
    
    s1 = (c1-c2)/3
    s1 = getQ(q,y,2.235)
    
    s2 = getQ(q,y,2.2563)
    s2 = getQ(q,y,2.285)

    fig,ax=plt.subplots(2,1,sharex=True)
#   plt.subplots_adjust(hspace=0)
    
    ax[0].pcolormesh(t,q,y.T,cmap=cmap,vmin=clim[0],vmax=clim[1])
    #ax[1].set_yticks( tt, map(str,tt))
    ax[0].set_xlim(-5,50)
    ax[0].set_ylim(1,3.3)

    ax[1].plot(t,s1,"o-")
    ax[1].plot(t,s2,"o-")
    ax[1].set_xlabel("time (ps)")
   

def _myprint(s,*args,**kw):
    print(time.asctime(),s,*args,**kw)
 
def autoAnalysisScanFolder(force=False,nmin=75):
    timetosleep = 0.1
    while True:
        try:
             #_myprint("looking for files")
             folder = "/sf/bernina/config/com/data/scan_data"
             files = glob.glob(folder + "/scan*_JF1p5M*.h5")
             files.sort()
             azfiles = [f for f in files if f.find("_az_") > -1 ]
             files   = [f for f in files if f.find("_az_") == -1 ]
             files = [f for f in files if int(os.path.basename(f)[4:7]) >nmin ]
             #_myprint("found",len(files),"files")
             if len(files) == 0: timetosleep = 60
             for fname in files:
                 isthereoff = fname.replace(".h5","_az_off.h5") in azfiles
                 isthereon  = fname.replace(".h5","_az_on.h5") in azfiles
                 if not (isthereon & isthereoff) or force:
                     try:
                         jf.readJF(fname)
                         temp = jf.azav("all",laser="off",force=force)
                         temp = jf.azav("all",laser="on",force=force)
                     except (OSError,KeyError):
                         pass
        except KeyboardInterrupt:
            break
        print(time.asctime(),"Auto analysis running, ... waiting for files ...")
        try:
            time.sleep(timetosleep)
            timetosleep = 60
        except KeyboardInterrupt:
            break

WP_scan1 = dict()
WP_scan1[100e-12]= 96
WP_scan1[300e-12]= 105
WP_scan1[10e-9]  = 95
WP_scan1[100e-9] = 98
WP_scan1[10e-6]  = 102
WP_scan1[100e-6] = 103

WP_scan2 = dict()
WP_scan2[1e-6]     = 166
WP_scan2[100e-9]   = 167
WP_scan2[10e-9]    = 168
WP_scan2[1e-9]     = 169
WP_scan2[100e-12]  = 170
WP_scan2[10e-6]    = 171
WP_scan2[10e-12]   = 173

def WPscans(scans = WP_scan1,what='integral'):

    if not what in ('integral','absolute'):
        print('Keyword argument "what" has to be integral or absolute')
    times = list(scans.keys())
    times.sort()

    results = [anaScan(scans[t],plot='none') for t in times]
    peaks   = [integrate_peaks(res,plot=False) for res in results]

    nscans = len(times)

    fig,axes=plt.subplots(nscans,1,sharey='col',sharex=True,figsize=[4,8])
    for ax,delay,res in zip(axes,times,results):
        for idiff in range(len(res.diffs)):
            ax.plot(res.q,res.diffs[idiff],label="%.2f mJ"%res.scan[idiff])
        ax.set_title("%s (scan%d)"%(trx.utils.timeToStr(delay),scans[delay]))
  

    #### PLOT PARS VS E #### 
    peak_ranges = peaks[0].peaks_pos
    peak_info = list(peaks[0].peak_info)
    fig,axes=plt.subplots(len(peak_ranges),1,sharey=False,sharex=True,figsize=[5,9])

    for ipeak,(ax,peak) in enumerate(zip(axes,peak_ranges)):
        ax.set_ylabel( peak_info[ipeak] )
        ax.set_title(str(peak))
        for res,delay in zip(peaks,times):
            ax.plot(res.scan,res[what][ipeak],"-o",label=trx.utils.timeToStr(delay))
    axes[-1].set_xlabel("E (mJ)")
    plt.legend()


    #### PLOT PARS VS t #### 
    E = results[0].scan
    res_integral = np.asarray( [p.integral for p in peaks if len(p.scan) == len(E) ] )
    res_absolute = np.asarray( [p.absolute for p in peaks if len(p.scan) == len(E) ] )
    fig,axes=plt.subplots(len(peak_ranges),2,sharex=True,sharey='col',figsize=[5,9])
    for ipeak,(ax,peak) in enumerate(zip(axes,peak_ranges)):
        ax[0].set_ylabel( peak_info[ipeak] )
        for ie,e in enumerate(E):
            ax[0].plot( times,res_integral[ipeak,:,ie],"o-",label="%.2f mJ"%e)
            ax[1].plot( times,res_absolute[ipeak,:,ie],"o-" )
    axes[-1,0].legend()
    axes[0][0].set_title("integral")
    axes[0][1].set_title("absolute")
#        ax[ipeak].plot( time, 
        

    return
    peak_ranges = peaks[0].peaks_pos
    peak_info = list(peaks[0].peak_info)
    fig,axes=plt.subplots(len(peak_ranges),1,sharey=False,sharex=True,figsize=[5,9])

    for ipeak,(ax,peak) in enumerate(zip(axes,peak_ranges)):
        ax.set_ylabel( peak_info[ipeak] )
        ax.set_title(str(peak))
        for res,delay in zip(peaks,times):
            ax.plot(res.scan,res[what][ipeak],"-o",label=trx.utils.timeToStr(delay))
    axes[-1].set_xlabel("E (mJ)")
    plt.legend()

def doruns():
    runs = (160,161,162,166,167,168,169,170,171,173)
    runs = (173,)
    for run in runs:
        if run == 160:
            refine = 0
        else:
            refine = 2
        anaScan(run,refine_pars=refine,force=True,peakThreshold=0.5,plot='none')

 
#if  __name__ == "__main__":
#    import sys 
#    autoAnalysis(sys.argv[1])
