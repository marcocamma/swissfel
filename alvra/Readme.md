# Overview

These scripts have been written during the p17807 and p17808 experiments in May 2019.

# ETOF analysis

To analyze the THz timing tool data using the BSREAD data using the photodiag.PalmSetup script
- first of all use modify the file etof_analysis.py, in particular the energy range, the PALM calibration file and the calibration factor (eV to fs) and the stage offset have to be adjusted
- then
```python
from swissfel.alvra.etof_analysis import do_all_may_files
do_all_may_files()
```

# YAG scan

To analyze the YAG scan

```python
from swissfel.alvra import yag_scans
yag_scans.ana_select(n=None,folder=json_folder,as_time=False,fit_with_tau=False)
```
It will fit the YAG transmission scan with and without timing tool
folder is the folder with the json metadata of the scan
fit_with_tau fits with an exponential convolved with a gaussian


# Drift Monitor

correct the common stage (globalglobi) using the THz timing tool
It has to be started from a machine that has access to the PVs of the THz analysis (that has to run in background). During the p1780[7|8] we used the sarop11-cpcl-palm118
```bash
python AlvraDrift_noEcode_dead_simple.py
```

