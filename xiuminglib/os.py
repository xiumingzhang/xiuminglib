import os
from os.path import abspath, join, exists, isdir
from shutil import rmtree
from glob import glob

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def sortglob(directory, filename='*', ext=None, ext_ignore_case=False):
    """Globs and then sorts according to a pattern ending in multiple extensions.

    Args:
        directory (str): Directory to glob, e.g., ``'/path/to/'``.
        filename (str or tuple(str), optional): Filename pattern excluding extensions, e.g., ``'img*'``.
        ext (str or tuple(str), optional): Extensions of interest, e.g., ``('png', 'PNG')``. ``None``
            means no extension, useful for files with no extension or folders.
        ext_ignore_case (bool, optional): Whether to ignore case for extensions.

    Returns:
        list(str): Sorted list of files globbed.
    """
    if ext is None:
        ext = ()
    elif isinstance(ext, str):
        ext = (ext,)
    if isinstance(filename, str):
        filename = (filename,)
    ext_list = []
    for x in ext:
        if not x.startswith('.'):
            x = '.' + x
        if ext_ignore_case:
            ext_list += [x.lower(), x.upper()]
        else:
            ext_list.append(x)
    files = []
    for f in filename:
        if ext_list:
            for e in ext_list:
                files += glob(join(directory, f + e))
        else:
            files += glob(join(directory, f))
    files_sorted = sorted(files)
    return files_sorted


def rmglob(path_pattern, exclude_dir=True):
    """Globs a pattern and then deletes the matches.

    Args:
        path_pattern (str): Pattern to glob, e.g., ``'/path/to/img???.png'``.
        exclude_dir (bool, optional): Whether to exclude directories from being deleted.
    """
    for x in glob(path_pattern):
        if isdir(x):
            if not exclude_dir:
                rmtree(x)
        else:
            os.remove(x)


def makedirs(directory, rm_if_exists=False):
    """Wraps :func:`os.makedirs` to support removing the directory if it already exists.

    Args:
        directory (str)
        rm_if_exists (bool, optional): Whether to remove the directory (and its contents)
            if it already exists.
    """
    logger_name = thisfile + '->makedirs()'

    if exists(directory):
        if rm_if_exists:
            logger.name = logger_name
            logger.info("Removed and then remade: %s", directory)
            rmtree(directory)
            os.makedirs(directory, exist_ok=True)
    else:
        os.makedirs(directory, exist_ok=True)


def fix_terminal():
    """Fixes messed up terminal."""
    from shlex import split
    from subprocess import Popen, DEVNULL

    cmd = 'stty sane'
    child = Popen(split(cmd), stdout=DEVNULL, stderr=DEVNULL)
    _, _ = child.communicate()
