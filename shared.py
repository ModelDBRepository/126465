from brian import *
from brian.hears import *
from brian.tools import datamanager
import os, sys, time, pickle

# shared samplerate, we use this for everything resampling if necessary
samplerate = 44.1*kHz
set_default_samplerate(samplerate)

# base path for data, and derived DataManager class which uses it
datapath, _ = os.path.split(__file__)
datapath = os.path.normpath(os.path.join(datapath, './data'))

# convenience function to get the IRCAM database, replace the file path
# with the location you downloaded it to.
def get_ircam():
    ircam_locations = [
        r'D:\HRTF\IRCAM',
        ]
    
    for path in ircam_locations:
        if os.path.exists(path):
            break
    else:
        raise IOError('Cannot find IRCAM HRTF location, add to ircam_locations in shared.py')
    ircam = IRCAM_LISTEN(path)
    return ircam
