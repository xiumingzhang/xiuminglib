from os.path import dirname, join

__all__ = [
    'blender',
    'camera',
    'config',
    'decorators',
    'general',
    'geometry',
    'image_processing',
    'io',
    'linear_algebra',
    'signal_processing',
    'tracker',
    'visualization',
]
from . import * # noqa: F403 # pylint: disable=wildcard-import


# ------ Constants

lib_dir = dirname(__file__)
data_dir = join(lib_dir, '..', 'data')

constants = {
    # Paths to data
    'checker_path': join(data_dir, 'texture/checker.png'),
}
