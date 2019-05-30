import numpy as np
import h5py 
import re
import collections
import os
import logging as log

import sys
sys.path.append( os.path.dirname(__file__) )

wdir =  '/sf/bernina/data/res/p16582/'
# Files for PBPS 138 and PIPS

tcalfile = os.path.join(wdir,'calib/wd136/wd136-5120.tcal')
vcalfile = os.path.join(wdir,'calib/wd136/wd136-5120.vcal')
#tcalfile = os.path.join(wdir,'calib/wd136/wd136-2008.tcal')
#vcalfile = os.path.join(wdir,'calib/wd136/wd136-2008.vcal')

try:
    import cal
    digitizer_cal = cal.Calib(calib_fina=vcalfile,t_calib_fina=tcalfile)
except ImportError:
    print("Could not import calibration routines for FE digitizer")
    digitizer_cal = None


try:
  from datastorage import DataStorage
  _has_datastorage = True
except ImportError:
  _has_datastorage = False

det_regex = re.compile("(.+):(\w+)-(.+)")

def reshapeToBroadcast(what,ref):
  """ expand the 1d array 'what' to allow broadbasting to match 
      multidimentional array 'ref'. The two arrays have to same the same 
      dimensions along the first axis
  """
  if what.shape == ref.shape: return what
  assert what.shape[0] == ref.shape[0],\
    "automatic reshaping requires same first dimention"
  shape  = [ref.shape[0],] + [1,]*(ref.ndim-1)
  return what.reshape(shape)


def _h5group_to_det(h5group):
    """ convert SwissFEL data group to Detector """
    data = h5group["data"]
    timestamp = h5group["timestamp"][:] + h5group["timestamp_offset"][:]/1e9
    pulse_id  = h5group["pulse_id"][:]
    additional_data = dict( timestamp_sec = h5group["timestamp"][:], timestamp_ns = h5group["timestamp_offset"][:] )
    return data,timestamp,pulse_id,additional_data


class Detector(object):
    def __init__(self,data,pulse_id=None,timestamp=None,readInMemory=False,additional_data=None):
        """ data can be an h5dataset if a h5group it assumed to be in the Swissfel format"""

        # check input type
        if isinstance(data,h5py.Group):
            data,timestamp,pulse_id,additional_data = _h5group_to_det(data)

        # if asked, actually read the detector
        if readInMemory: data = data[:]
        self.dtype = data.dtype

        # sanity check
        if pulse_id is not None and pulse_id.shape[0] != data.shape[0]:
          log.warn("Warning, shape of pulse_id and data did not match (data shape = %s, pulse_id shape = %s)" % \
          (data.shape,pulse_id.shape))
        if timestamp is not None and timestamp.shape[0] != data.shape[0]:
          log.warn("Warning, shape of timestamp and data did not match (data shape = %s, pulse_id shape = %s)" % \
          (data.shape,pulse_id.shape))
        
        self.data = data
        self.timestamp = timestamp
        self.pulse_id = pulse_id
        if additional_data is not None:
            for key,value in additional_data.items(): setattr(self,key,value)
        self.shotsize = self.dtype.itemsize
        for ndim in self.data.shape[1:]: self.shotsize *= ndim

    def getShots(self,shotslice): return self.data[shotslice]

    def __getitem__(self,idx): return self.data[idx]

    def __repr__(self):
        return "Detector object, shape %s, type %s" % (self.data.shape,self.dtype)




class Diode(Detector):

    def __init__(self,data,pulse_id=None,timestamp=None,additional_data=None,\
        ringBufferInfo=None,channel=None):

        Detector.__init__(self,data,pulse_id=pulse_id,timestamp=timestamp,\
           readInMemory=True,additional_data=additional_data)

        self.ringBufferInfo = Detector(ringBufferInfo) if ringBufferInfo is \
                              not None else None

        self.channel = int(channel)
        self.process()

    def process(self,nPointsDark=80,invert=False,calibrate=True,subtract_baseline=False):

        if digitizer_cal is not None and self.ringBufferInfo is not None and calibrate:
            # save raw data
            self.rawdata = self.data
            self.data,self.tdata = digitizer_cal.calibrate_raw_hdf5(self.data,\
               self.ringBufferInfo.data,channel=self.channel)
        else:
            self.tdata = None

        baselines = self.data[:,:nPointsDark].mean(axis=1)
        if subtract_baseline: self.data = self.data-baselines[:,np.newaxis]
        self.sum  = self.data.sum(axis=1)
        if self.tdata is not None:
            self.integral = np.trapz( self.data, self.tdata )
        else:
            self.integral = np.trapz( self.data )
        if invert: self.intensity *= -1

    def __repr__(self):
        if hasattr(self,"intensity"):
            return "Diode object %d shots, values %s" % (self.data.shape[0],str(self.intensity[:10]))
        else:
            return "Diode object %d shots" % (self.data.shape[0])
        


class HDF5_swissfel(object):
    def __init__(self,fname,autofind=True):
      self.fname = fname
      self.h5handle = h5py.File(fname,"r") 
      if autofind: self.find_dets()

    def find_dets(self,cleanup_names=True,readInMemory=False):
        """ based on the assumption source:detname-[....] """
        dets = DataStorage() if _has_datastorage else dict()
        for name in self.h5handle.keys():
            match = det_regex.search(name)
            if match is None: continue
            source,detname,field = match.groups()
            if cleanup_names:
              source  = source.replace("-","_")
              detname = detname.replace("-","_")
              field   = field.replace("-","_")
            if source not in dets: dets[source] = dict()
            if detname not in dets[source]: dets[source][detname] = dict()
            #print(field)
            if source.find("PBPS")>-1 and field == "DATA":
            #if field.find("DATA_CALIBRATED")>-1:
                #print("diode",name)
                # detname is something like Lnk9Ch15
                channel = detname.split("Ch")[1]
                ring_buffer = name.replace("DATA","DRS_TC")
                det = Diode(self.h5handle[name],ringBufferInfo=self.h5handle[ring_buffer],
                      channel=channel)
            elif source.find("PBPS")>-1 and field.find("DATA_CALIBRATED")>-1:
                channel = detname.split("Ch")[1]
                det = Diode(self.h5handle[name],channel=channel)
            else:
                #print("det",name)
                det = Detector(self.h5handle[name],readInMemory=readInMemory)
            dets[source][detname][field] = det
        for k,v in dets.items(): setattr(self,k,v)
        self.dets = dets

    def __repr__(self):
        if hasattr(self,"dets"):
            return str(self.dets)
        else:
            return "HDF5_swissfel dataset"
        


def example_one_file():
  d = HDF5_swissfel("diode_01.h5")
  print("reading in memory")
  dets = d.find_dets(readInMemory=True)
  print(dets["SF_DIGITIZER_01"]["Lnk9Ch14"])
  print("returning links")
  dets = d.find_dets(readInMemory=False)
  print(dets["SF_DIGITIZER_01"]["Lnk9Ch14"])
