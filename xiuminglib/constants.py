from os import environ
from os.path import abspath, join, dirname


class Dir:
    tmp = environ.get('TMP_DIR', '/tmp/')
    mstatus = join(environ.get('MSTATUS_BACKEND_DIR', '/tmp/machine-status'), 'runtime')
    data = abspath(join(dirname(__file__), '..', 'data'))


class Path:
    # Textures
    checker = join(Dir.data, 'textures/checker.png')

    # Images
    cameraman = join(Dir.data, 'images/cameraman.png')
    lenna = join(Dir.data, 'images/lenna.png')

    # Models
    armadillo = join(Dir.data, 'models/armadillo.ply')
    buddha = join(Dir.data, 'models/buddha.ply')
    bunny = join(Dir.data, 'models/bunny.ply')
    dragon = join(Dir.data, 'models/dragon.ply')
    teapot = join(Dir.data, 'models/teapot.obj')
