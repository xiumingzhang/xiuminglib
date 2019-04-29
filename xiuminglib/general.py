import sys
import os
from os.path import abspath, join, exists, dirname, isdir
from shutil import rmtree
from glob import glob

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def print_attrs(obj, excerpts=None, excerpt_win_size=60, max_recursion_depth=None):
    """Prints all attributes, recursively, of an object.

    Args:
        obj (object): Object in which we search for the attribute.
        excerpts (str or list(str), optional): Print only excerpts containing
            certain attributes. ``None`` means to print all.
        excerpt_win_size (int, optional): How many characters get printed around a match.
        max_recursion_depth (int, optional): Maximum recursion depth. ``None`` means no limit.
    """
    import re
    import jsonpickle
    import yaml

    logger_name = thisfile + '->print_attrs()'

    if isinstance(excerpts, str):
        excerpts = [excerpts]
    assert isinstance(excerpts, list) or excerpts is None

    try:
        serialized = jsonpickle.encode(obj, max_depth=max_recursion_depth)
    except RecursionError as e:
        logger.name = logger_name
        logger.error("RecursionError: %s! Please specify a limit to retry",
                     str(e))
        sys.exit(1)

    if excerpts is None:
        # Print all attributes
        logger.name = logger_name
        logger.info("All attributes:")
        print(yaml.dump(yaml.load(serialized), indent=4))
    else:
        for x in excerpts:
            # For each attribute of interest, print excerpts containing it
            logger.name = logger_name
            logger.info("Excerpt(s) containing '%s':", x)

            mis = [m.start() for m in re.finditer(x, serialized)]
            if not mis:
                logger.name = logger_name
                logger.info("%s: No matches! Retry maybe with deeper recursion")
            else:
                for mii, mi in enumerate(mis):
                    # For each excerpt
                    m_start = mi - excerpt_win_size // 2
                    m_end = mi + excerpt_win_size // 2
                    print(
                        "Match %09d (index: %09d): ...%s\033[0;31m%s\033[00m%s..." % (
                            mii,
                            mi,
                            serialized[m_start:mi],
                            serialized[mi:(mi + len(x))],
                            serialized[(mi + len(x)):m_end],
                        )
                    )


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


def ask_to_proceed(msg, level='warning'):
    """Pauses there to ask the user whether to proceed.

    Args:
        msg (str): Message to display to the user.
        level (str, optional): Message level, essentially deciding the message color:
            ``'info'``, ``'warning'``, or ``'error'``.
    """
    logger_name = thisfile + '->ask_to_proceed()'
    logger_print = getattr(logger, level)
    logger.name = logger_name
    logger_print(msg)
    logger_print("Proceed? (y/n)")
    need_input = True
    while need_input:
        response = input().lower()
        if response in ('y', 'n'):
            need_input = False
        if need_input:
            logger.name = logger_name
            logger.error("Enter only y or n!")
    if response == 'n':
        sys.exit()


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


def load_or_save(data_f, fallback=None):
    """Loads the data file if it exists. Otherwise, if fallback is provided,
    call fallback and save its return to disk.

    Args:
        data_f (str): Path to the data file, whose extension will be used for deciding
            how to load the data.
        fallback (function, optional): Fallback function used if data file doesn't exist.
            Its return will be saved to ``data_f`` for future loadings. It should not
            take arguments, but if yours requires taking arguments, just wrap yours with::

                fallback=lambda: your_fancy_func(var0, var1)

    Raises:
        NotImplementedError: If file extension is neither .npy nor .npz.

    Returns:
        Data loaded if ``data_f`` exists; otherwise, ``fallback``'s return
        (``None`` if no fallback).

    Writes:
        - Return by the fallback, if provided.
    """
    import numpy as np

    logger_name = thisfile + '->load_or_save()'

    # Decide data file type
    ext = data_f.split('.')[-1].lower()
    if ext == 'npy':
        load_func = np.load
        save_func = np.save
    elif ext == 'npz':
        load_func = np.load
        save_func = np.savez
    else:
        raise NotImplementedError(ext)

    # Load or call fallback
    if exists(data_f):
        data = load_func(data_f)
        msg = "Loaded: "
    else:
        msg = "Non-existent, "
        if fallback is None:
            data = None
            msg += "and fallback not provided: "
        else:
            data = fallback()
            out_dir = dirname(data_f)
            makedirs(out_dir)
            save_func(data_f, data)
            msg += "but called fallback and saved its return: "
    msg += data_f

    logger.name = logger_name
    logger.info(msg)
    return data


def fix_terminal():
    """Fixes messed up terminal."""
    from shlex import split
    from subprocess import Popen, DEVNULL

    cmd = 'stty sane'
    child = Popen(split(cmd), stdout=DEVNULL, stderr=DEVNULL)
    _, _ = child.communicate()
