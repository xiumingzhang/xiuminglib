import os
from os.path import abspath, join, exists, isdir
from shutil import rmtree
from glob import glob

from . import config
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


def makedirs(directory, rm_if_exists=False, google=False):
    """Wraps :func:`os.makedirs` to support removing the directory if it already exists.

    Args:
        directory (str)
        rm_if_exists (bool, optional): Whether to remove the directory (and its contents)
            if it already exists.
        google (bool, optional): Whether on Google's infra.
    """
    logger_name = thisfile + '->makedirs()'

    if google:
        from google3.pyglib import gfile
        exists_func = gfile.Exists
        delete_func = gfile.DeleteRecursively
        mkdir_func = gfile.MakeDirs
    else:
        exists_func = exists
        delete_func = rmtree
        mkdir_func = os.makedirs

    if exists_func(directory):
        if rm_if_exists:
            logger.name = logger_name
            logger.info("Removed and then remade: %s", directory)
            delete_func(directory)
            mkdir_func(directory)
    else:
        mkdir_func(directory)


def make_exp_dir(directory, param_dict, rm_if_exists=False):
    """Makes an experiment output folder by hashing the experiment parameters.

    Args:
        directory (str): The made folder will be under this.
        param_dict (dict): Dictionary of the parameters identifying the experiment.
            It is sorted by its keys, so different orders lead to the same hash.
        rm_if_exists (bool, optional): Whether to remove the experiment folder
            if it already exists.

    Writes
        - The experiment parameters in ``<directory>/<hash>/param.json``.

    Returns:
        str: The experiment output folder just made.
    """
    from collections import OrderedDict
    from json import dump

    logger_name = thisfile + '->make_exp_dir()'

    hash_seed = os.environ.get('PYTHONHASHSEED', None)
    if hash_seed != '0':
        logger.name = logger_name
        logger.warning(("PYTHONHASHSEED is not 0, so the same param_dict has different hashes "
                        "across sessions. Consider disabling this randomization with "
                        "`PYTHONHASHSEED=0 python your_script.py`"))

    param_dict = OrderedDict(sorted(param_dict.items()))
    param_hash = str(hash(str(param_dict)))
    assert param_hash != '' # gotta be careful because of rm_if_exists

    directory = join(directory, param_hash)
    makedirs(directory, rm_if_exists=rm_if_exists)

    # Write parameters into a .json
    json_f = join(directory, 'param.json')
    with open(json_f, 'w') as h:
        dump(param_dict, h, indent=4, sort_keys=True)

    logger.name = logger_name
    logger.info("Parameters dumped to: %s", json_f)

    return directory


def fix_terminal():
    """Fixes messed up terminal."""
    from shlex import split
    from subprocess import Popen, DEVNULL

    cmd = 'stty sane'
    child = Popen(split(cmd), stdout=DEVNULL, stderr=DEVNULL)
    _, _ = child.communicate()
