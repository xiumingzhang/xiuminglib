from os import environ
from os.path import dirname, join, abspath
from .config import get_all
lib_dir = dirname(__file__)
__all__ = get_all(lib_dir)
from . import * # noqa: F401,F403 # pylint: disable=wildcard-import


# ------ Constants

data_dir = abspath(join(lib_dir, '..', 'data'))

constants = {
    'dir_tmp': environ.get('TMP_DIR', '/tmp/'),
    'dir_mstatus': join(environ.get('MSTATUS_BACKEND_DIR', '/tmp/'), 'runtime'),
    # Paths to data
    'path_checker': join(data_dir, 'texture/checker.png'),
    'path_cameraman': join(data_dir, 'images/cameraman_grayscale.png'),
}
