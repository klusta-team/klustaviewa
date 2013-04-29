"""User preferences."""
# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
import logging

# Logging level, can be DEBUG, INFO or WARNING.
loglevel = logging.DEBUG

# -----------------------------------------------------------------------------
# Waveform view
# -----------------------------------------------------------------------------
# Approximate maximum number of spikes to show. Should be
# about 100 for low-end graphics cards, 1000 for high-end ones.
waveforms_nspikes_max_expected = 100

# The minimum number of spikes per cluster to display.
waveforms_nspikes_per_cluster_min = 3


# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
# Delay between two successive automatic operations in unit tests for views.
test_operator_delay = .01

# Whether to automatically close the views during unit testing.
test_auto_close = True


