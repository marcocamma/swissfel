import numpy as np

def subtractReferences(i,idx_ref, useRatio = False):
  """ given data in i (first index is shot num) and the indeces of the 
      references (idx_ref, array of integers) it interpolates the closest
      reference data for each shot and subtracts it (or divides it, depending
      on useRatio = [True|False]; 
      Note: it works in place (i.e. it modifies i) """
  iref=np.empty_like(i)
  idx_ref = np.squeeze(idx_ref)
  idx_ref = np.atleast_1d(idx_ref)
  # sometime there is just one reference (e.g. sample scans)
  if idx_ref.shape[0] == 1:
    if useRatio:
      return i/i[idx_ref]
    else:
      return i-i[idx_ref]
  # references before first ref are "first ref"
  iref[:idx_ref[0]] = i[idx_ref[0]]
  # references after last ref are "last ref"
  iref[idx_ref[-1]:] = i[idx_ref[-1]]
  _ref = 0
  for _i in range(idx_ref[0],idx_ref[-1]):
    if _i in idx_ref: continue
    idx_ref_before = idx_ref[_ref]
    idx_ref_after  = idx_ref[_ref+1]
    ref_before = i[idx_ref_before]
    ref_after  = i[idx_ref_after]
    weight_before = float(_i-idx_ref_before)/(idx_ref_after-idx_ref_before)
    weight_after  = 1-weight_before
    # normal reference for an on chi, the weighted average
    iref[_i]      = weight_before*ref_before + weight_after*ref_after
    if _i>=idx_ref_after-1: _ref += 1
  # take care of the reference for the references ...
  if len(idx_ref) >  2:
    iref[idx_ref[0]] = i[idx_ref[1]]
    iref[idx_ref[-1]] = i[idx_ref[-2]]
    for _i in range(1,len(idx_ref)-1):
      idx_ref_before = idx_ref[_i-1]
      idx_ref_after  = idx_ref[_i+1]
      ref_before = i[idx_ref_before]
      ref_after  = i[idx_ref_after]
      weight_before = float(idx_ref[_i]-idx_ref_before)/(idx_ref_after-idx_ref_before)
      weight_after  = 1-weight_before
      # normal reference for an on chi, the weighted average
      iref[idx_ref[_i]]    = weight_before*ref_before + weight_after*ref_after
  else:
    #print(idx_ref)
    #print(iref[idx_ref])
    iref[idx_ref]=i[idx_ref[0]]
    #print(iref[idx_ref])
  if useRatio:
    i /= iref
  else:
    i -= iref
  return i
