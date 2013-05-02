"""User preferences."""
# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
import logging

# Logging level, can be DEBUG, INFO or WARNING.
loglevel = logging.DEBUG


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
# Should the software ask the user to save upon closing?
prompt_save_on_exit = True


# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------
# Approximate maximum number of spikes to show. Should be
# about 100 for low-end graphics cards, 1000 for high-end ones.
waveforms_nspikes_max_expected = 100

# The minimum number of spikes per cluster to display.
waveforms_nspikes_per_cluster_min = 3


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------
# Opacity value of the background spikes.
feature_background_alpha = .2

# Maximum number of spikes in the view.
features_nspikes_background_max = 10000
features_nspikes_selection_max = 1000
features_nspikes_per_cluster_min = 100
                    
                    
# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
# Delay between two successive automatic operations in unit tests for views.
test_operator_delay = .01

# Whether to automatically close the views during unit testing.
test_auto_close = True


