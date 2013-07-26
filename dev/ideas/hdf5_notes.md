Notes about the new HDF5 file format
====================================

26/07/2013

I'm currently writing a conversion tool from the Klusters files to the new HDF5 file format.

  * The implemented format in this tool is slightly different from the one currently implemented in SpikeDetekt. Eventually, SpikeDetekt will be updated to integrate those changes, but in the meantime, the HDF5 files created by SpikeDetekt are automatically deleted by default at the end of the SpikeDetekt script. This way, there will be no conflicts between the SpikeDetekt HDF5 files and the new ones
  
  * The new file format is meant to be flexible, i.e. it can have optional fields. Someone reading those files should check that a particular attribute or column exists before accessing it. For instance, when converting from the old Klusters format to the new one, it may happen that there is no mask file. In this case, we don't create a mask column in the table (it would be a waste of memory to fill it with a default value).

Decisions that need to be made
------------------------------

  * Do we need 1 or 2 fields for "mask"? We decided "mask_binary" and "mask_float" but I'm not sure whether it's a good idea. It is redundant information as we can trivially (in Python or in any language) retrieve the binary value from the float value. 
  
  * Should we fix the data type of the fields, or should we let them be flexible? We could retrieve the actual data types when loading the files.
  
  * Which data type for the floating mask? I'd say uint8 (1 byte), from 0 to 255.

  * Which data type for the "time" (in number of samples)? I'd say uint64, as uint32 can become limited to a few hours with high sampling rates.
  
  * Which data type for the features? For now it is int16, except for the last column (time) which would need to be int64. That means that if we want to have the time in the features field, it needs to be int64. The other possibility is to get rid of the last column with time and keep int16.
  
  * What about putting a version number in the HDF5 files metadata? The file format might evolve while we're working on it, and putting a version number will help us avoid incompatibilities later.
  
  
  