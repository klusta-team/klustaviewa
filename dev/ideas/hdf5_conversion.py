import os
import tables

import numpy as np
import matplotlib.pyplot as plt

import klustaviewa.dataio as kvio

# Get the filenames.
folder = r"D:\Spike sorting\137_34_shankA_29cat_shabnam"
basename = "137_34_shankA_29cat_shabnam"
filenames = kvio.find_filenames(os.path.join(folder, basename + '.fet.1'))
fileindex = kvio.find_index(os.path.join(folder, basename + '.fet.1'))

# Open small Klusters files.
clusters = kvio.read_clusters(filenames['clu'])
# aclusters = kvio.read_clusters(filenames['aclu'])
# acluinfo = kvio.read_cluster_info(filenames['acluinfo'])
# groupinfo = kvio.read_group_info(filenames['groupinfo'])
metadata = kvio.read_xml(filenames['xml'], fileindex)
# probe = kvio.read_probe(filenames['probe'])

nspikes = len(clusters)
nchannels = metadata['nchannels']
nsamples = metadata['nsamples']

# Open big Klusters files.
fetfile = kvio.MemMappedText(filenames['fet'], np.int16, skiprows=1)
spkfile = kvio.MemMappedBinary(filenames['spk'], np.int16, rowsize=nchannels * nsamples)
    
# Create the HDF5 file.
hdf_main = tables.openFile(basename + '.main.h5', mode='w')
hdf_main.createGroup('/', 'shanks')
hdf_main.createGroup('/shanks', 'shank0')

hdf_waves = tables.openFile(basename + '.waves.h5', mode='w')
hdf_waves.createGroup('/', 'shanks')
hdf_waves.createGroup('/shanks', 'shank0')

spikes_description = dict(time=tables.Int64Col(),
                          mask_binary=tables.BoolCol(shape=(metadata['nchannels'],)),
                          mask_float=tables.Int8Col(shape=(metadata['nchannels'],)),
                          features=tables.Int16Col(shape=(metadata['fetdim'] * metadata['nchannels'] + 1,)),
                          cluster=tables.Int32Col(),)
waves_description = dict(
        wave=tables.Int16Col(shape=(metadata['nsamples'], metadata['nchannels'])),
        wave_unfiltered=tables.Int16Col(shape=(metadata['nsamples'], metadata['nchannels'])))
spikes_table = hdf_main.createTable('/shanks/shank0', 'spikes', spikes_description)
waves_table = hdf_waves.createTable('/shanks/shank0', 'waves', waves_description)

for spike in xrange(10):
    fet = fetfile.next()
    spk = spkfile.next().reshape((nsamples, nchannels))

    row_main = spikes_table.row
    row_wave = waves_table.row

    row_main['time'] = fet[-1]
    # row_main['mask_binary'] = fet[-1]
    # row_main['mask_float'] = fet[-1]
    row_main['features'] = fet
    row_main['cluster'] = clusters[spike]
    row_main.append()
    
    row_wave['wave'] = spk


hdf_main.flush()
hdf_main.close()

hdf_waves.flush()
hdf_waves.close()

