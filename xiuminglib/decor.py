"""Decorators that wrap a function.

If the function is defined in the file where you want to use the decorator,
you can decorate the function at define time:

.. code-block:: python

    @decorator
    def somefunc():
        return

If the function is defined somewhere else, do:

.. code-block:: python

    from numpy import mean

    mean = decorator(mean)
"""

from time import time, sleep
from os import makedirs, environ
import os.path
from os.path import abspath, join, dirname, basename, getmtime

from .os import call, _is_cnspath

from .config import create_logger
logger, thisfile = create_logger(abspath(__file__))


def colossus_interface(somefunc):
    """Wraps black-box functions to read from and write to Google Colossus.

    Because it's hard (if possible at all) to figure out which path is
    input, and which is output, when the input function is black-box, this is
    a "best-effort" decorator with heuristics (see below for warnings).

    Warning:
        It's easy to identify a CNS path, but not so a local path. The
        heuristic here is to consider any string containing ``'/'`` that
        doesn't start with ``'/cns/'`` as a local path. This could be wrong
        of course.

    This decorator works by looping through all the positional and keyword
    parameters, copying CNS paths that exist prior to ``somefunc`` execuation
    to temporary local locations, running ``somefunc`` and writing its output
    to local locations, and finally copying local paths that get modified by
    ``somefunc`` to their corresponding CNS locations.

    Warning:
        Therefore, if ``somefunc``'s output already exists (e.g., you are
        re-running the function to overwrite the old result), it will be
        copied to local, overwritten by ``somefunc`` locally, and finally
        copied back to CNS. This doesn't lead to wrong behaviors, but is
        inefficient.

    This decorator doesn't depend on Blaze, as it's using the ``fileutil``
    CLI, rather than ``google3.pyglib.gfile``. This is convenient in at least
    two cases:

    - You are too lazy to use Blaze, want to run tests quickly on your local
      machine, but need access to CNS files.
    - Your IO is more complex than what ``with gfile.Open(...) as h:`` can do
      (e.g., a Blender function importing an object from a path), in which
      case you have to copy the CNS file to local ("local" here could also
      mean a Borglet's local).

    This interface generally works with resolved paths (e.g.,
    ``/path/to/file``), but not with wildcard paths (e.g., ``/path/to/???``),
    sicne it's hard (if possible at all) to guess what your function tries to
    do with such wildcard paths.

    Writes
        - Input files copied from Colossus to ``$TMP/``.
        - Output files generated to ``$TMP/``, to be copied to Colossus.
    """
    logger_name = thisfile + '->@colossus_interface(%s())' \
        % somefunc.__name__

    def could_be_path(x):
        # FIXME: not-so-elegant heuristic
        return isinstance(x, str) and '/' in x

    # $TMP set by Borg or yourself (e.g., with .bashrc)
    tmp_dir = environ.get('TMP', '/tmp/')

    def get_cns_info(cns_path):
        # Existence; file or directory
        testf, _, _ = call('fileutil test -f %s' % cns_path)
        testd, _, _ = call('fileutil test -d %s' % cns_path)
        if testf == 1 and testd == 1:
            exists = False
            isdir = False
        elif testf == 1 and testd == 0:
            exists = True
            isdir = True
        elif testf == 0 and testd == 1:
            exists = True
            isdir = False
        else:
            raise NotImplementedError("What does this even mean?")
        # Deal with '/'-ending paths
        if cns_path.endswith('/'):
            assert isdir, "Not a directory, but ends with '/'?"
            cns_path = cns_path[:-1]
            assert not cns_path.endswith('/'), "Path shouldn't end with '//'"
        # Path guaranteed not to end with '/', so that basename is not ''
        local_path = join(tmp_dir, '%f_%s' % (time(), basename(cns_path)))
        return cns_path, exists, isdir, local_path

    def get_local_info(local_path):
        exists = os.path.exists(local_path)
        isdir = os.path.isdir(local_path)
        # Deal with '/'-ending paths
        if local_path.endswith('/'):
            assert isdir, "Not a directory, but ends with '/'?"
            local_path = local_path[:-1]
            assert not local_path.endswith('/')
        return local_path, exists, isdir

    def cp(src, dst, isdir=False):
        parallel_copy = 10
        cmd = 'fileutil cp -f -colossus_parallel_copy'
        if isdir:
            cmd += ' -R -parallel_copy=%d %s' % \
                (parallel_copy, join(src, '*'))
        else:
            cmd += ' %s' % src
        cmd += ' %s' % dst
        logger.name = logger_name
        if call(cmd)[0] == 0:
            logger.info("\n%s\n\tcopied to\n%s", src, dst)
        else:
            logger.warning("\n%s\n\tfailed to be copied to\n%s", src, dst)

    def wrapper(*arg, **kwargs):
        # Fetch info. for all CNS paths
        arg_local, kwargs_local = [], {}
        cns_info, local_info = {}, {}
        # Positional arguments
        for x in arg:
            if _is_cnspath(x):
                cns_path, cns_exists, cns_isdir, local_path = \
                    get_cns_info(x)
                cns_info[cns_path] = (cns_exists, cns_isdir, local_path)
                arg_local.append(local_path)
            elif could_be_path(x): # don't touch non-CNS files
                local_path, local_exists, local_isdir = get_local_info(x)
                local_info[local_path] = (local_exists, local_isdir)
                arg_local.append(local_path)
            else: # intact
                arg_local.append(x)
        # Keyword arguments
        for k, v in kwargs.items():
            if _is_cnspath(v):
                cns_path, cns_exists, cns_isdir, local_path = \
                    get_cns_info(v)
                cns_info[cns_path] = (cns_exists, cns_isdir, local_path)
                kwargs_local[k] = local_path
            elif could_be_path(v): # don't touch non-CNS files
                local_path, local_exists, local_isdir = get_local_info(v)
                local_info[local_path] = (local_exists, local_isdir)
                kwargs_local[k] = local_path
            else: # intact
                kwargs_local[k] = v
        # For reading: copy CNS paths that exist to local
        # TODO: what if some of those paths are not input? Copying them to
        # local is a waste
        for cns_path, (cns_exists, cns_isdir, local_path) in \
                cns_info.items():
            if cns_exists:
                cp(cns_path, local_path, isdir=cns_isdir)
        # Run the real function
        t0 = time()
        results = somefunc(*arg_local, **kwargs_local)
        # For writing: copy local paths that are just modified and correspond
        # to CNS paths back to CNS
        for cns_path, (cns_exists, cns_isdir, local_path) in \
                cns_info.items():
            if os.path.exists(local_path) and getmtime(local_path) > t0:
                cp(local_path, cns_path, isdir=cns_isdir)
        return results

    return wrapper


def timeit(somefunc):
    """Outputs the time a function takes to execute."""
    logger_name = thisfile + '->@timeit(%s())' % somefunc.__name__

    def wrapper(*arg, **kwargs):
        t0 = time()
        results = somefunc(*arg, **kwargs)
        t = time() - t0
        logger.name = logger_name
        logger.info("Time elapsed: %f seconds", t)
        return results

    return wrapper


def existok(makedirs_func):
    """Implements the ``exist_ok`` flag in 3.2+, which avoids race conditions,
    where one parallel worker checks the folder doesn't exist and wants to
    create it with another worker doing so faster.
    """
    logger_name = thisfile + '->@existok(%s())' % makedirs_func.__name__

    def wrapper(*args, **kwargs):
        try:
            makedirs_func(*args, **kwargs)
        except OSError as e:
            if e.errno != 17:
                raise
            logger.name = logger_name
            logger.debug("%s already exists, but that is OK", args[0])

    return wrapper


def main():
    """Unit tests that can also serve as example usage."""
    # timeit
    @timeit
    def findsums(x, y, z):
        sleep(1)
        return x + y, x + z, y + z, x + y + z
    print(findsums(1, 2, 3))

    # existok
    newdir = join(dirname(__file__), 'test')
    makedirs_ = existok(makedirs)
    makedirs_(newdir)
    makedirs_(newdir)


if __name__ == '__main__':
    main()
