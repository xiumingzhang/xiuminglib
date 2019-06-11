from os import environ
from os.path import abspath, join, dirname


dir_tmp = environ.get('TMP_DIR', '/tmp/')
dir_mstatus = join(environ.get('MSTATUS_BACKEND_DIR', '/tmp/'), 'runtime')

# Paths to data
data_dir = abspath(join(dirname(__file__), '..', 'data'))
path_checker = join(data_dir, 'texture/checker.png')
path_cameraman = join(data_dir, 'images/cameraman_grayscale.png')
