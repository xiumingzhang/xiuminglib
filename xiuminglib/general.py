import sys
from os import makedirs
from os.path import abspath, join, exists, dirname
import re
from glob import glob
import numpy as np

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def print_attrs(obj, excerpts=None, excerpt_win_size=60, max_recursion_depth=None):
    """
    Print all attributes, recursively, of an object

    Args:
        obj: Object in which we search for the attribute
            Any object
        excerpts: Print only excerpts containing certain attributes
            A single string or a list thereof
            Optional; defaults to None (print all)
        excerpt_win_size: How many characters get printed around a match
            Positive integer
            Optional; defaults to 60
        max_recursion_depth: Maximum recursion depth
            Positive integer
            Optional; defaults to None (no limit)
    """
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
                        "Match %d (index: %d): ...%s\033[0;31m%s\033[00m%s..." % (
                            mii,
                            mi,
                            serialized[m_start:mi],
                            serialized[mi:(mi + len(x))],
                            serialized[(mi + len(x)):m_end],
                        )
                    )


def sortglob(directory, filename, exts, ext_ignore_case=False):
    """
    Glob and then sort according to the pattern ending in multiple extensions

    Args:
        directory: Directory to glob
            String; e.g., '/path/to/'
        filename: Filename pattern excluding extensions
            String; e.g., 'batch000_*'
        exts: Extensions of interest
            Set of strings; e.g., ('.png', '.PNG')
        ext_ignore_case: Whether to ignore case for extensions
            Boolean
            Optional; defaults to False

    Returns:
        files_sorted: Sorted list of files globbed
            List of strings
    """
    ext_list = []
    for ext in exts:
        if not ext.startswith('.'):
            ext = '.' + ext
        if ext_ignore_case:
            ext_list += [ext.lower(), ext.upper()]
        else:
            ext_list.append(ext)
    files = []
    for ext in ext_list:
        files += glob(join(directory, filename + ext))
    files_sorted = sorted(files)
    return files_sorted


def ask_to_proceed(msg, level='warning'):
    """
    Pause there to ask the user whether to proceed

    Args:
        msg: Message to display to the user
            String
        level: Message level, essentially deciding the message color
            'info' | 'warning' | 'error'
            Optional; defaults to 'warning'
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


def load_or_save(data_f, fallback=None):
    """
    Load the data file if it exists. Otherwise, if fallback provided,
        call fallback and save its return

    Args:
        data_f: Path to the data file, whose extension will be used for deciding
            how to load the data
            String
        fallback: Fallback function if data file doesn't exist, whose return will
            be saved to <data_f> for future use
            function that doesn't take arguments. Can easily construct one with
                `fallback=lambda: your_fancy_func(var0, var1)`
            Optional; defaults to None

    Returns:
        data: Data loaded if existent; otherwise, fallback's return or
            None if fallback is not provided
    """
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
            if not exists(out_dir):
                makedirs(out_dir)
            save_func(data_f, data)
            msg += "but called fallback and saved its return: "
    msg += data_f

    logger.name = logger_name
    logger.info(msg)
    return data
