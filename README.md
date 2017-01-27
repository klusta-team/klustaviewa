# KlustaViewa

KlustaViewa is a graphical interface for the manual stage of spike sorting.

**NOTE**: this legacy project is superseded by the phy KwikGUI that comes with the [KlustaSuite](https://github.com/kwikteam/klusta).


## Installation

Installing KlustaViewa may be tricky because it relies on very old dependencies (numpy 1.8, pandas 0.12, pyqt 4...).

1. Make sure that you have [**miniconda**](http://conda.pydata.org/miniconda.html) installed. You can choose the Python 3.5 64-bit version for your operating system (Linux, Windows, or OS X).
2. **Download the environment file:**
    * [Linux](https://raw.githubusercontent.com/klusta-team/klustaviewa/master/installer/environment-linux.yml)
    * [OS X](https://raw.githubusercontent.com/klusta-team/klustaviewa/master/installer/environment-osx.yml)
    * [Windows](https://raw.githubusercontent.com/klusta-team/klustaviewa/master/installer/environment-win.yml)
3. Open a terminal (on Windows, `cmd`, not Powershell) in the directory where you saved the file and type:

    ```bash
    conda install conda=3  # SKIP THIS LINE ON WINDOWS
    conda env create -n klustaviewa -f environment-XXX.yml  # replace `XXX` by your system
    source activate klustaviewa  # omit the `source` on Windows
    ```

4. **Done**! Now, to use KlustaViewa, you have to first type `source activate klustaviewa` in a terminal (omit the `source` on Windows), and then call `klustaviewa yourfile.kwik`.


## Documentation

There is a [user guide here](https://github.com/klusta-team/klustaviewa/blob/master/docs/manual.md) but it is quite old, so some instructions refer to a previous version.


## Credits

**KlustaViewa** is developed by [Cyrille Rossant](http://cyrille.rossant.net), [Max Hunter](https://iris.ucl.ac.uk/iris/browse/profile?upi=MLDHU99), and [Kenneth Harris](https://iris.ucl.ac.uk/iris/browse/profile?upi=KDHAR02), in the [Cortexlab](https://www.ucl.ac.uk/cortexlab), University College London.
