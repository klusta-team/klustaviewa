"""User preferences for KlustaViewa."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import numpy as np


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

# Console logging level, can be DEBUG, INFO or WARNING.
loglevel = logging.INFO

# Level of the logging file. DEBUG, INFO or WARNING, or just None to disable.
loglevel_file = logging.DEBUG


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
# Should the software ask the user to save upon closing?
prompt_save_on_exit = True
delay_timer = .05
delay_buffer = .1


# -----------------------------------------------------------------------------
# Similarity matrix
# -----------------------------------------------------------------------------
similarity_measure = 'gaussian'  # or 'kl' for KL divergence


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------
# Approximate maximum number of spikes to show. Should be
# about 100 for low-end graphics cards, 1000 for high-end ones.
waveforms_nspikes_max_expected = 100

# The minimum number of spikes per cluster to display.
waveforms_nspikes_per_cluster_min = 10


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------
# Opacity value of the background spikes.
feature_background_alpha = .25

# Opacity value of the spikes in the selected clusters.
feature_selected_alpha = .75

# Number of spikes to show in the background.
features_nspikes_background_max = 10000  

# Maximum number of spikes to show in the selected clusters.
###########
# WARNING #
###########
# Do not change this value, otherwise you will have problems when
# splitting clusters (unselected spikes).
features_nspikes_selection_max = np.inf  # always show all selected clusters

# Minimum number of spikes to show per selected cluster.
###########
# WARNING #
###########
# Do not change this value, otherwise you will have problems when
# splitting clusters (unselected spikes).
features_nspikes_per_cluster_min = np.inf  # always show all selected clusters

# Unit of the spike time in the feature view. Can be 'samples' or 'second'.
features_info_time_unit = 'second'


# -----------------------------------------------------------------------------
# Correlograms view
# -----------------------------------------------------------------------------
# Maximum number of clusters to show in the correlograms view.
correlograms_max_nclusters = 20


# -----------------------------------------------------------------------------
# IPython import path
# -----------------------------------------------------------------------------
# Paths where all .py files are loaded in IPython view.
# "~" corresponds to the user home path, C:\Users\Username\ on Windows,
# /home/username/ on Linux, etc.
ipython_import_paths = ['~/.klustaviewa/code']


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
# Delay between two successive automatic operations in unit tests for views.
test_operator_delay = .1

# Whether to automatically close the views during unit testing.
test_auto_close = True


