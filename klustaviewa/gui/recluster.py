"""Launching script."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import os
import sys
import os.path as op
import tempfile
from subprocess import Popen
import threading
import argparse

import numpy as np
from kwiklib import (Experiment, get_params, load_probe, create_files, 
    read_raw, Probe, convert_dtype, read_clusters,
    files_exist, add_clustering, delete_files, exception)

PARAMS_KK = dict(
    MaskStarts = 10,
    #MinClusters = 100 ,
    #MaxClusters = 110,
    MaxPossibleClusters =  500,
    FullStepEvery = 10,
    MaxIter = 100,
    RandomSeed =  654,
    Debug = 0,
    SplitFirst = 20 ,
    SplitEvery =    0 ,
    PenaltyK = 1,
    PenaltyKLogN = 0,
    Subset = 1,
    PriorPoint = 1,
    SaveSorted = 0,
    SaveCovarianceMeans = 0,
    UseMaskedInitialConditions = 1,
    AssignToFirstClosestMask = 1,
    UseDistributional = 1,
)

def write_mask(mask, filename, fmt="%f"):
    with open(filename, 'w') as fd:
        fd.write(str(mask.shape[1])+'\n') # number of features
        np.savetxt(fd, mask, fmt=fmt)

def write_fet(fet, filepath):
    with open(filepath, 'w') as fd:
        #header line: number of features
        fd.write('%i\n' % fet.shape[1])
        #next lines: one feature vector per line
        np.savetxt(fd, fet, fmt="%i")

def save_old(exp, shank, spikes, dir=None):
    chg = exp.channel_groups[shank]
            
    # Create files in the old format (FET and FMASK)
    fet = chg.spikes.features_masks[spikes, ...]
    if fet.ndim == 3:
        masks = fet[:,:,1]  # (nsamples, nfet)
        fet = fet[:,:,0]  # (nsamples, nfet)
    else:
        masks = None
    res = chg.spikes.time_samples[spikes]
    
    times = np.expand_dims(res, axis =1)
    masktimezeros = np.zeros_like(times)
    
    fet = convert_dtype(fet, np.int16)
    fet = np.concatenate((fet, times),axis = 1)
    mainfetfile = os.path.join(dir, exp.name + '.fet.' + str(shank))
    write_fet(fet, mainfetfile)

    if masks is not None:
        fmasks = np.concatenate((masks, masktimezeros),axis = 1)
        fmaskfile = os.path.join(dir, exp.name + '.fmask.' + str(shank))
        write_mask(fmasks, fmaskfile, fmt='%f')
    
def run_klustakwik(exp, channel_group=None, clusters=None, **kwargs):
    name = exp.name
    
    # Set the KlustaKwik parameters.
    params = PARAMS_KK.copy()
    for key in PARAMS_KK.keys():
        # Update the PARAMS_KK keys if they are specified directly
        # but ignore the kwargs keys that do not appear in PARAMS_KK.
        params[key] = kwargs.get(key.lower(), params[key])
        
    # Switch to temporary directory.
    start_dir = os.getcwd()
    experiment_dir = exp.dir or os.path.dirname(exp._filenames['kwik'])
    tmpdir = os.path.join(experiment_dir, '_recluster')
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    os.chdir(tmpdir)
    
    shank = channel_group

    # Find the spikes belonging to the clusters to recluster.
    spikes = np.nonzero(np.in1d(exp.channel_groups[shank].spikes.clusters.main[:], clusters))[0]
    
    save_old(exp, shank, spikes, dir=tmpdir)
    
    # Generate the command for running klustakwik.
    # TODO: add USERPREF to specify the full path to klustakwik 
    cmd = ['KlustaKwik', name, str(shank)]
    for key, val in params.iteritems():
        cmd += ['-' + str(key), str(val) ]
    
    # Run KlustaKwik.
    p = Popen(cmd)
    p.wait()
    
    # Read back the clusters.
    clu = read_clusters(name + '.clu.' + str(shank))

    # DEBUG
    # clu = np.random.randint(low=1000, high=1030, size=len(spikes))

    # Switch back to original dir.
    os.chdir(start_dir)
        
    return spikes, clu
