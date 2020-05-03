import logging
from platform import system


def create_logger(file_abspath, path_starts_from='xiuminglib'):
    """Creates a logger for functions in the library.

    Args:
        file_abspath (str): Absolute path to the file that uses the logger.
        path_starts_from (str, optional): Truncates ``thisfile`` so that it
            starts from this.

    Returns:
        tuple:
            - **logger** (*logging.Logger*) -- Logger created.
            - **thisfile** (*str*) -- Partial path to the user file (e.g.,
              starting from package name).
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    folder_names = file_abspath.split('/')
    if path_starts_from in folder_names:
        start_idx = folder_names.index(path_starts_from)
    else:
        start_idx = 0
    thisfile = '/'.join(folder_names[start_idx:])

    return logger, thisfile


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
    raise NotImplementedError(
        "This library has yet to be made Windows-compatible")

# All non-Windows platforms are supporting ANSI escapes so we use them
logging.StreamHandler.emit = _add_coloring_to_emit_ansi(
    logging.StreamHandler.emit)
