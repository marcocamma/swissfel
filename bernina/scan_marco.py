import numpy as np
import collections
import matplotlib.pyplot as plt
from joblib import Memory
import time
folder = "/tmp"
mycache = Memory(folder)

_ret = collections.namedtuple("analyzedData","no x y label")


analyzed = dict()

def mytime(f,*args):
  def ff(*args):
     t0 = time.time()
     r = f(*args)
     print(time.time()-t0)
     return r
  return ff



@mytime
def analyze1(scannu,label="ciao"):
  global analyzed
  if not scannu in analyzed:
    time.sleep(0.1)
    print("recalculating")
    x = np.random.random(10)
    y = x**2
    analyzed[scannu] = _ret(no=scannu,x=x,label=label,y=y)
  return analyzed[scannu]

@mycache.cache
def analyze2(scannu,label="ciao"):
    print("recalculating")
    x = np.random.random(10)
    y = x**3
    z = 3
    return dict(no=scannu,x=x,label=label,y=y)


  
def plot( scanno ):
    scanno = list(scanno)
    scanno.sort()
    for scan in scanno:
        res = analyze(scan)
        plt.plot(res.x,res.y,label="%s %s"%(res.no,res.label))
