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
        return ('%(asctime)s,%(msecs)03d  %(levelname)-7s  %(message)s')
    else:
        return '%(asctime)s  %(message)s'

def get_caller():
    tb = traceback.extract_stack()[-5]
    module = os.path.splitext(os.path.basename(tb[0]))[0]#.ljust(24)
    line = str(tb[1])#.ljust(4)
    caller = "{0:s}:{1:s}".format(module, line)
    return caller.ljust(24)

def get_var_info(name, var):
    return name + ": Type = " + str(type(var)) + ", Value = " + str(var)

def debugvar(name, var):
    debug(get_var_info(name, var))
    

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
    def __init__(self, fmt=None, stream=None, level=None, name=None,
            print_caller=True, handler=None):
        if stream is None:
            stream = sys.stdout
        if name is None:
            name = APPNAME
        self.name = name
        self.print_caller = print_caller
        if handler is None:
            self.stream = stream
            self.handler = logging.StreamHandler(self.stream)
        else:
            self.handler = handler
        self.level = level
        self.fmt = fmt
        # Set the level and corresponding formatter.
        self.set_level(level, fmt)
        
    def set_level(self, level=None, fmt=None):
        # Default level and format.
        if level is None:
            level = self.level or logging.INFO
        if fmt is None:
            fmt = self.fmt or get_log_format(level == logging.DEBUG)
        # Create the Logger object.
        self._logger = logging.getLogger(self.name)
        # Create the formatter.
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        self.handler.setFormatter(formatter)
        # Configure the logger.
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._logger.addHandler(self.handler)
        
    def get_message(self, msg):
        msg = str(msg)
        if self.print_caller:
            return get_caller() + msg
        else:
            return msg
        
    def debug(self, msg):
        self._logger.debug(self.get_message(msg))
        
    def info(self, msg):
        self._logger.info(self.get_message(msg))
        
    def warn(self, msg):
        self._logger.warn(self.get_message(msg))
        
    def exception(self, msg):
        self._logger.exception(self.get_message(msg))
        

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


class FileLogger(Logger):
    def __init__(self, filename=None, **kwargs):
        kwargs['handler'] = logging.FileHandler(filename)
        super(FileLogger, self).__init__(**kwargs)
        
    def close(self):
        self.handler.close()
        self._logger.removeHandler(self.handler)
        del self.handler
        del self._logger


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
LOGGERS = {}

def register(logger):
    name = logger.name
    if name not in LOGGERS:
        LOGGERS[name] = logger

def unregister(logger):
    name = logger.name
    if name in LOGGERS:
        LOGGERS[name].close()
        del LOGGERS[name]
        
# Console logger.
LOGGER = ConsoleLogger(name='{0:s}.console'.format(APPNAME))
register(LOGGER)

def debug(msg):
    for name, logger in LOGGERS.iteritems():
        logger.debug(msg)

def info(msg):
    for name, logger in LOGGERS.iteritems():
        logger.info(msg)

def warn(msg):
    for name, logger in LOGGERS.iteritems():
        logger.warn(msg)

def exception(msg):
    for name, logger in LOGGERS.iteritems():
        logger.exception(msg)

def set_level(msg):
    for name, logger in LOGGERS.iteritems():
        logger.set_level(msg)


# Capture all exceptions.
def handle_exception(exc_type, exc_value, exc_traceback):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    exception(msg)
sys.excepthook = handle_exception

