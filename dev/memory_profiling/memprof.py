import os
import numpy as np
import pandas as pd
import memory_profiler
import gc
import inspect
from operator import itemgetter
from itertools import groupby
from pympler.asizeof import asizeof

mid = lambda x: (x.__array_interface__['data'][0] 
        if isinstance(x, np.ndarray) else id(x))

def getsize(o):
    if isinstance(o, np.ndarray):
        return o.nbytes
    elif isinstance(o, (pd.DataFrame, pd.Panel)):
        return getsize(o.values)
    elif isinstance(o, (list, tuple)):
        return np.sum((getsize(oo) for oo in o))
    else:
        return asizeof(o)

def fmtsize(nbytes):
    return "{0:.3f} MB".format(nbytes / (1024. ** 2))

def profile_mem(obj):
    sizes = [(name, getsize(getattr(obj, name)), mid(getattr(obj, name))) 
        for name in dir(obj) if not inspect.ismethod(getattr(obj, name))]
    # Remove the attributes pointing to the same object.
    sizes = sorted(sizes, key=itemgetter(2))
    sizes = [list(v)[0] for k, v in groupby(sizes, itemgetter(2))]
    # Sort the sizes by decreasing order.
    sizes = sorted(sizes, key=itemgetter(1))[::-1]

    print "TOTAL SIZE:", fmtsize(np.sum(map(itemgetter(1), sizes)))
    print
    for (name, size, x) in sizes:
        if size >= 1024:
            print "{0:s}\t{1:s}".format(fmtsize(size), name)


