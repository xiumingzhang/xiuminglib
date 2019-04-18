"""Library setup including setting logging colors, etc."""

from os.path import join, basename, isdir
from glob import glob
import logging
from platform import system

logging_warn = logging.WARN
# so that won't need to import a package just for its constants


def get_all(lib_dir):
    # Figure out subpackages and modules
    modules = []
    for f in sorted(glob(join(lib_dir, '*'))):
        base = basename(f)
        if not base.endswith('.pyc') and base != '__init__.py' and base != '__pycache__':
            if base.endswith('.py'):
                base = base[:-3]
            else:
                assert isdir(f), "Neither a module (.py) nor a subpackage (folder): %s" % f
            modules.append(base)
    return modules


def create_logger(file_abspath, level=logging.INFO):
    """Creates a logger for functions in the library.

    Args:
        file_abspath (str): Absolute path to the file that uses the logger.
        level (int, optional): Logging level. Defaults to ``logging.INFO``.

    Returns:
        tuple:
            - **logger** (*logging.Logger*) -- Logger created.
            - **thisfile** (*str*) -- Partial path to the user file, starting from package name.
    """
    starting_from = 'xiuminglib'
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    folder_names = file_abspath.split('/')
    if starting_from in folder_names:
        start_idx = folder_names.index(starting_from)
    else:
        start_idx = 0
    thisfile = '/'.join(folder_names[start_idx:])
    return logger, thisfile


# ---------------------------- Logging Colors


def _add_coloring_to_emit_windows(fn):
    # add methods we need to the class
    def _out_handle(self):
        return windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
    # out_handle = property(_out_handle)

    def _set_color(self, code):
        # Constants from the Windows API
        self.STD_OUTPUT_HANDLE = -11
        hdl = windll.kernel32.GetStdHandle(self.STD_OUTPUT_HANDLE)
        windll.kernel32.SetConsoleTextAttribute(hdl, code)

    setattr(logging.StreamHandler, '_set_color', _set_color)

    def new(*args):
        FOREGROUND_BLUE = 0x0001 # text color contains blue
        FOREGROUND_GREEN = 0x0002 # text color contains green
        FOREGROUND_RED = 0x0004 # text color contains red
        FOREGROUND_INTENSITY = 0x0008 # text color is intensified
        FOREGROUND_WHITE = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED
        # winbase.h
        STD_INPUT_HANDLE = -10
        STD_OUTPUT_HANDLE = -11
        STD_ERROR_HANDLE = -12
        # wincon.h
        FOREGROUND_BLACK = 0x0000
        FOREGROUND_BLUE = 0x0001
        FOREGROUND_GREEN = 0x0002
        FOREGROUND_CYAN = 0x0003
        FOREGROUND_RED = 0x0004
        FOREGROUND_MAGENTA = 0x0005
        FOREGROUND_YELLOW = 0x0006
        FOREGROUND_GREY = 0x0007
        FOREGROUND_INTENSITY = 0x0008 # foreground color is intensified
        BACKGROUND_BLACK = 0x0000
        BACKGROUND_BLUE = 0x0010
        BACKGROUND_GREEN = 0x0020
        BACKGROUND_CYAN = 0x0030
        BACKGROUND_RED = 0x0040
        BACKGROUND_MAGENTA = 0x0050
        BACKGROUND_YELLOW = 0x0060
        BACKGROUND_GREY = 0x0070
        BACKGROUND_INTENSITY = 0x0080 # background color is intensified
        levelno = args[1].levelno
        if levelno >= 50:
            color = BACKGROUND_YELLOW | FOREGROUND_RED | FOREGROUND_INTENSITY | BACKGROUND_INTENSITY
        elif levelno >= 40:
            color = FOREGROUND_RED | FOREGROUND_INTENSITY
        elif levelno >= 30:
            color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
        elif levelno >= 20:
            color = FOREGROUND_GREEN
        elif levelno >= 10:
            color = FOREGROUND_MAGENTA
        else:
            color = FOREGROUND_WHITE
        args[0]._set_color(color)
        ret = fn(*args)
        args[0]._set_color(FOREGROUND_WHITE)
        return ret

    return new


def _add_coloring_to_emit_ansi(fn):
    # Add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if levelno >= 50:
            color = '\x1b[31m' # red
        elif levelno >= 40:
            color = '\x1b[31m' # red
        elif levelno >= 30:
            color = '\x1b[33m' # yellow
        elif levelno >= 20:
            color = '\x1b[32m' # green
        elif levelno >= 10:
            color = '\x1b[35m' # pink
        else:
            color = '\x1b[0m' # normal
        args[1].msg = color + args[1].msg + '\x1b[0m' # normal
        return fn(*args)
    return new


if system() == 'Windows':
    # Windows does not support ANSI escapes and we are using API calls to set the console color
    from ctypes import windll
    logging.StreamHandler.emit = _add_coloring_to_emit_windows(
        logging.StreamHandler.emit)
else:
    # All non-Windows platforms are supporting ANSI escapes so we use them
    logging.StreamHandler.emit = _add_coloring_to_emit_ansi(
        logging.StreamHandler.emit)
