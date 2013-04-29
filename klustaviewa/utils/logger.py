"""Logger utility classes and functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging
import traceback

from klustaviewa.utils.globalpaths import APPNAME


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_log_format(debug=False):
    if debug:
        # return ('%(asctime)s,%(msecs)03d  %(levelname)-7s  P:%(process)-4d  '
                # 'T:%(thread)-4d  %(message)s')
        return ('%(asctime)s,%(msecs)03d  %(levelname)-7s  %(message)s')
    else:
        return '%(asctime)s  %(message)s'

def get_caller():
    # return ""
    tb = traceback.extract_stack()[-3]
    module = os.path.splitext(os.path.basename(tb[0]))[0].ljust(24)
    line = str(tb[1]).ljust(4)
    return "L:%s  %s" % (line, module)


# -----------------------------------------------------------------------------
# Stream classes
# -----------------------------------------------------------------------------
class StringStream(object):
    """Logger stream used to store all logs in a string."""
    def __init__(self):
        self.string = ""
        
    def write(self, line):
        self.string += line
        
    def flush(self):
        pass
        
    def __repr__(self):
        return self.string
        
        
# -----------------------------------------------------------------------------
# Logging classes
# -----------------------------------------------------------------------------
class Logger(object):
    """Save logging information to a stream."""
    def __init__(self, fmt=None, stream=None, level=None):
        if stream is None:
            stream = sys.stdout
        self.stream = stream
        self.handler = logging.StreamHandler(self.stream)
        # Set the level and corresponding formatter.
        self.set_level(level, fmt)
        
    def set_level(self, level=None, fmt=None):
        # Default level and format.
        if level is None:
            level = logging.INFO
        if fmt is None:
            fmt = get_log_format(level == logging.DEBUG)
        # Create the Logger object.
        self._logger = logging.getLogger(APPNAME)
        # Create the formatter.
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        self.handler.setFormatter(formatter)
        # Configure the logger.
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._logger.addHandler(self.handler)
        
    def debug(self, msg):
        self._logger.debug(get_caller() + msg)
        
    def info(self, msg):
        self._logger.info(get_caller() + msg)
        
    def warn(self, msg):
        self._logger.warn(get_caller() + msg)
        
    def exception(self, msg):
        self._logger.exception(get_caller() + msg)


class StringLogger(Logger):
    def __init__(self, **kwargs):
        kwargs['stream'] = StringStream()
        super(StringLogger, self).__init__(**kwargs)
        
    def __repr__(self):
        return self.stream.__repr__()


class ConsoleLogger(Logger):
    def __init__(self, **kwargs):
        kwargs['stream'] = sys.stdout
        super(ConsoleLogger, self).__init__(**kwargs)


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
LOGGER = ConsoleLogger()
debug = LOGGER.debug
info = LOGGER.info
warn = LOGGER.warn
exception = LOGGER.exception
set_level = LOGGER.set_level
