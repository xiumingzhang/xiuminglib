"""Library setup including setting logging colors, etc."""

from os.path import join, basename, isdir
from glob import glob
import logging
from platform import system

logging_warn = logging.WARN
# so that won't need to import a package just for its constants


def create_logger(file_abspath, level=logging.INFO,
                  path_starts_from='xiuminglib'):
    """Creates a logger for functions in the library.

    Args:
        file_abspath (str): Absolute path to the file that uses the logger.
        level (int, optional): Logging level.
        path_starts_from (str, optional): Truncates ``thisfile`` so that it
            starts from this.

    Returns:
        tuple:
            - **logger** (*logging.Logger*) -- Logger created.
            - **thisfile** (*str*) -- Partial path to the user file (e.g.,
              starting from package name).
    """
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    folder_names = file_abspath.split('/')
    if path_starts_from in folder_names:
        start_idx = folder_names.index(path_starts_from)
    else:
        start_idx = 0
    thisfile = '/'.join(folder_names[start_idx:])
    return logger, thisfile


def to_import_at_init(lib_dir, incl_subpkg=True):
    """Figures out what modules (and maybe also subpackages) to import in
    __init__().
    """
    all_list = []
    for f in sorted(glob(join(lib_dir, '*'))):
        base = basename(f)
        if not base.endswith('.pyc') and \
                base != '__init__.py' and \
                base != '__pycache__':
            if base.endswith('.py'):
                # Modules for sure will be imported
                base = base[:-3]
                all_list.append(base)
            else:
                assert isdir(f), \
                    "Neither a module (.py) nor a subpackage (folder): %s" \
                    % f
                # Subpackages are to be imported only if asked
                if incl_subpkg:
                    all_list.append(base)
    return all_list


# ---------------------------- Logging Colors


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
