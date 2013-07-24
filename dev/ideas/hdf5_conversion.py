import os
import tables

import numpy as np
import matplotlib.pyplot as plt

import klustaviewa.dataio as kvio

# Get the filenames.
folder = r"D:\Spike sorting\test"
basename = "test"
filenames = kvio.find_filenames(os.path.join(folder, basename))
fileindex = kvio.find_index(os.path.join(folder, basename))

hdf_main_filename = os.path.join(folder, basename + '.main.h5')
hdf_wave_filename = os.path.join(folder, basename + '.wave.h5')

# Open small Klusters files.
data = {}
metadata = kvio.read_xml(filenames['xml'], fileindex)
data['clu'] = kvio.read_clusters(filenames['clu'])
if 'aclu' in filenames and os.path.exists(filenames['aclu']):
    data['aclu'] = kvio.read_clusters(filenames['aclu'])
if 'acluinfo' in filenames and os.path.exists(filenames['acluinfo']):
    data['acluinfo'] = kvio.read_cluster_info(filenames['acluinfo'])
if 'groupinfo' in filenames and os.path.exists(filenames['groupinfo']):
    data['groupinfo'] = kvio.read_group_info(filenames['groupinfo'])
if 'probe' in filenames:
    data['probe'] = kvio.read_probe(filenames['probe'])

# Find out the number of columns in the .fet file.
f = open(filenames['fet'], 'r')
f.readline()
fetcol = len(f.readline().split(' '))


nspikes = len(data['clu'])
nchannels = metadata['nchannels']
nsamples = metadata['nsamples']

# Open big Klusters files.
data['fet'] = kvio.MemMappedText(filenames['fet'], np.int16, skiprows=1)
data['spk'] = kvio.MemMappedBinary(filenames['spk'], np.int16, rowsize=nchannels * nsamples)
if 'mask' in filenames and os.path.exists(filenames['mask']):
    data['mask'] = kvio.MemMappedText(filenames['mask'], np.float32, skiprows=1)
    hasmask = True
else:
    hasmask = False
    
# Create the HDF5 file.
hdf_main = tables.openFile(hdf_main_filename, mode='w')
hdf_main.createGroup('/', 'shanks')
hdf_main.createGroup('/shanks', 'shank0')

hdf_waves = tables.openFile(hdf_wave_filename, mode='w')
hdf_waves.createGroup('/', 'shanks')
hdf_waves.createGroup('/shanks', 'shank0')

spikes_description = dict(
    time=tables.Int64Col(),
    mask_binary=tables.BoolCol(shape=(fetcol,)),
    mask_float=tables.Int8Col(shape=(fetcol,)),
    features=tables.Int16Col(shape=(fetcol,)),
    cluster=tables.Int32Col(),)
waves_description = dict(
        wave=tables.Int16Col(shape=(metadata['nsamples'], metadata['nchannels'])),
        wave_unfiltered=tables.Int16Col(shape=(metadata['nsamples'], metadata['nchannels'])))

spikes_table = hdf_main.createTable('/shanks/shank0', 'spikes', spikes_description)
waves_table = hdf_waves.createTable('/shanks/shank0', 'waves', waves_description)

for spike in xrange(nspikes):
    
    if spike % 1000 == 0:
        print "{0:.1f}%\r".format(spike / float(nspikes) * 100),

    fet = data['fet'].next()
    spk = data['spk'].next().reshape((nsamples, nchannels))
    
    if hasmask:
        mask = data['mask'].next()

    row_main = spikes_table.row
    row_wave = waves_table.row

    row_main['time'] = fet[-1]
    
    if hasmask:
        row_main['mask_binary'] = mask.astype(np.bool)
        row_main['mask_float'] = (mask * 255).astype(np.uint8)
    else:
        row_main['mask_binary'] = True
        row_main['mask_float'] = 255
        
    row_main['features'] = fet
    row_main['cluster'] = data['clu'][spike]
    
    row_wave['wave'] = spk

    row_main.append()
    row_wave.append()

hdf_main.flush()
hdf_main.close()

hdf_waves.flush()
hdf_waves.close()

