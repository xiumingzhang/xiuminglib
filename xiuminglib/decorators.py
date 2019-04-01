from time import time, sleep
from os import makedirs
from os.path import abspath, join, dirname

import config
logger, thisfile = config.create_logger(abspath(__file__))


def timeit(somefunc):
    """
    Outputs the time a function takes to execute
    """
    logger_name = thisfile + '->@timeit'

    def wrapper(*arg, **kwargs):
        funcname = somefunc.__name__
        logger.name = logger_name
        logger.info("Function %s started", funcname)
        t0 = time()
        results = somefunc(*arg, **kwargs)
        t = time() - t0
        logger.name = logger_name
        logger.info("    ... and done in %.2f seconds", t)
        return results

    return wrapper


def existok(makedirs_func):
    """
    Implements the exist_ok flag in >= 3.2, which avoids race conditions,
    where one parallel worker checks the folder doesn't exist and wants to
    create it with another worker doing so faster
    """
    logger_name = thisfile + '->@existok'

    def wrapper(*args, **kwargs):
        try:
            makedirs_func(*args, **kwargs)
        except OSError as e:
            if e.errno != 17:
                raise
            logger.name = logger_name
            logger.debug("%s already exists, but that is OK", args[0])
    return wrapper


# Tests
if __name__ == '__main__':
    # timeit
    @timeit
    def findsums(x, y, z):
        sleep(1)
        return x + y, x + z, y + z, x + y + z
    print(findsums(1, 2, 3))
    # existok
    newdir = join(dirname(__file__), 'test')
    makedirs = existok(makedirs)
    makedirs(newdir)
    makedirs(newdir)
