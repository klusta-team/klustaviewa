"""Launching script."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import os
import sys
import os.path as op
import tempfile
import argparse

import numpy as np
from kwiklib import (Experiment, get_params, load_probe, create_files, 
    read_raw, Probe, convert_dtype, read_clusters,
    files_exist, add_clustering, delete_files, exception)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _load_files_info(prm_filename, dir=None):
    dir_, filename = op.split(prm_filename)
    dir = dir or dir_
    basename, ext = op.splitext(filename)
    if ext == '':
        ext = '.prm'
    prm_filename = op.join(dir, basename + ext)
    assert op.exists(prm_filename)
    
    # Load PRM file.
    prm = get_params(prm_filename)
    nchannels = prm.get('nchannels')
    assert nchannels > 0
    
    # Find PRB path in PRM file, and load it.
    prb_filename = prm.get('prb_file')
    if not op.exists(prb_filename):
        prb_filename = op.join(dir, prb_filename)
    prb = load_probe(prb_filename)
        
    # Find raw data source.
    data = prm.get('raw_data_files')
    if isinstance(data, basestring):
        if data.endswith('.dat'):
            data = [data]
    if isinstance(data, list):
        for i in range(len(data)):
            if not op.exists(data[i]):
                data[i] = op.join(dir, data[i])
        
    experiment_name = prm.get('experiment_name')
    
    return dict(prm=prm, prb=prb, experiment_name=experiment_name, nchannels=nchannels,
                data=data, dir=dir)    
    
def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
    
def print_path():
    print '\n'.join(os.environ["PATH"].split(os.pathsep))
    
def check_path():
    prog = 'klustakwik'
    if not (which(prog) or which(prog + '.exe')):
        print("Error: '{0:s}' is not in your system PATH".format(prog))
        return False
    return True

# -----------------------------------------------------------------------------
# KlustaKwik
# -----------------------------------------------------------------------------
PARAMS_KK = dict(
    MaskStarts = 100,
    #MinClusters = 100 ,
    #MaxClusters = 110,
    MaxPossibleClusters =  500,
    FullStepEvery = 10,
    MaxIter = 10000,
    RandomSeed =  654,
    Debug = 0,
    SplitFirst = 20 ,
    SplitEvery = 100 ,
    PenaltyK = 0,
    PenaltyKLogN = 1,
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

def save_old(exp, shank, dir=None):
    chg = exp.channel_groups[shank]
            
    # Create files in the old format (FET and FMASK)
    fet = chg.spikes.features_masks[...]
    if fet.ndim == 3:
        masks = fet[:,:,1]  # (nsamples, nfet)
        fet = fet[:,:,0]  # (nsamples, nfet)
    else:
        masks = None
    res = chg.spikes.time_samples[:]
    
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
    
def run_klustakwik(exp, **kwargs):
    name = exp.name
    shanks = exp.channel_groups.keys()
    
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
    
    clus = {}
    for shank in shanks:
        # chg = exp.channel_groups[shank]
        
        save_old(exp, shank, dir=tmpdir)
        
        # Generate the command for running klustakwik.
        # TODO: add USERPREF to specify the full path to klustakwik 
        cmd = 'KlustaKwik {name} {shank} {params}'.format(
            name=name,
            shank=shank,
            params=' '.join(['-{key} {val}'.format(key=key, val=str(val))
                                for key, val in params.iteritems()]),
        )
        
        # Run KlustaKwik.
        os.system(cmd)
        
        # Read back the clusters.
        clu = read_clusters(name + '.clu.' + str(shank))
        
        # Put the clusters in the kwik file.
        #exp.channel_groups[shank].spikes.clusters.main[spikes_idx] = clu
        clus[shank] = clu

    # Switch back to original dir.
    os.chdir(start_dir)
        
    return clus
