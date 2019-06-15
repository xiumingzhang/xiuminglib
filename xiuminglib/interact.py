import sys
from os.path import abspath

from .config import create_logger
logger, thisfile = create_logger(abspath(__file__))


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


def format_print(msg, fmt):
    """Prints a message with format.

    Args:
        msg (str): Message to print.
        fmt (str): Format; try your luck with any value (if it's illegal, you will
            be prompted all legal values.

    Raises:
        ValueError: If the input format is illegal.
    """
    fmt_strs = {
        'header': '\033[95m',
        'okblue': '\033[94m',
        'okgreen': '\033[92m',
        'warn': '\033[93m',
        'fail': '\033[91m',
        'bold': '\033[1m',
        'underline': '\033[4m',
    }
    end_str = '\033[0m'
    if fmt in fmt_strs.keys():
        start_str = fmt_strs[fmt]
    else:
        raise ValueError("Legal values for fmt: %s, but input is '%s'"
                         % (list(fmt_strs.keys()), fmt))
    print(start_str + msg + end_str)
