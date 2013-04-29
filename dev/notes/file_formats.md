HDF5 format in SpikeDetekt
==========================

pytables format, which is compatible with h5py

### SpikeTable_temp table

Rows: one per spike

Columns:

  * channel_mask            int8            (Nchannels,)
  * clu                     int32           ()
  * fet                     float32         (Nchannels, fetdim)
  * fet_mask                int8            (Nchannels x fetdim + 1,)
  * float_channel_mask      float32         (Nchannels,)
  * float_fet_mask          float32         (Nchannels x fetdim + 1,)
  * time                    int32           ()
  * unfiltered_wave         int32           (Nsamples, Nchannels)
  * wave                    float32         (Nsamples, Nchannels)


### DatChannels Table

1 column, Nchannels rows of int64 with the index of the channel




File formats in klustaviewa
=====================

Notes from 12/09/2012 meeting
-----------------------------

  * HDF5 should be used for every file.
  
  * We'll start with home-made file formats, e.g. three files:
    * .h5dat (for instance), a HDF5 version of the .dat file with the raw data
    * .h5fil (for instance), a HDF5 version of the .fil file with the filtered
       data
    * .h5whatever, a HDF5 file with the contents of the old .spk, .res, .clu...
      files.

  * All code that is related to files should be gathered in a single Python
    module (IO module), that allows to load and save from and to the relevant
    file formats. There are several advantages to do that:
    
      * We can modify/add new file formats easily by adding new options for
        import and export (under the condition that there is a standard, common
        internal format for the Python arrays).
        
      * If a new file format emerges at some point, we would just need to 
        change this IO module, and not to look for dependencies with the old 
        formats everywhere in the code.
        
      * It's easier to maintain backward compatibility this way, since we may
        add the possibility to load from old file formats.

  * A possibility: one main HDF5 file with soft links to other HDF5 files
    (.dat.h5, .fil.h5, .spk.h5, etc.).

  
  
  
Current file formats
====================

DAT file
--------
Binary file, 16 bits integers (20 kHz).

Contains the raw trace.

Array format: cluster/sample


FIL file
--------
Like DAT but HP filtered (20 kHz).


EEG file
--------
Like DAT but LP filtered (1.25 kHz)


CLU file
--------
ASCII file.

Contains the clusters for each spike.

First line: number of clusters.
Next lines: cluster indices for each spike.


RES file
--------
ASCII file.

Contains the spike times (integers in unit of the time bin).

Each line contains the spike time, for each spike.


FET file
--------
ASCII file.

Contains the PCA features.

First line: number of dimensions.
Next lines: tab-separated numbers cluster/electrode, for each spike


SPK file
--------
Binary file, 16 bits integers.

Contains the filtered waveforms, for each spike.

Array format: electrode/sample/spike

