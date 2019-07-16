from os.path import abspath
from importlib import import_module

from .config import create_logger
logger, thisfile = create_logger(abspath(__file__))


def preset_import(module_name):
    """A unified importer for both regular and ``google3`` modules, according
    to specified presets/profiles (e.g., ignoring ``ModuleNotFoundError``).
    """
    if module_name == 'cv2':
        # "//third_party/py/cvx2",
        try:
            from cvx2 import latest as mod
        except ModuleNotFoundError:
            try:
                import cv2 as mod
            except ModuleNotFoundError:
                mod = None
        # TODO: use import_module_404ok()

    elif module_name == 'gfile':
        # "//pyglib:gfile",
        # "//file/colossus/cns",
        mod = import_module_404ok('gfile', package='google3.pyglib')

    elif module_name in ('bpy', 'bmesh', 'OpenEXR', 'Imath'):
        # "//third_party/py/Imath",
        # "//third_party/py/OpenEXR",
        mod = import_module_404ok(module_name)

    elif module_name in ('Vector', 'Matrix', 'Quaternion'):
        mod = import_module_404ok(module_name, package='mathutils')

    elif module_name == 'BVHTree':
        mod = import_module_404ok(module_name, package='mathutils.bvhtree')

    else:
        raise NotImplementedError(module_name)

    return mod


def import_module_404ok(*args, **kwargs):
    """Returns ``None`` (instead of failing) in the case of
    ``ModuleNotFoundError``.
    """
    logger_name = thisfile + '->import_module_404ok()'
    try:
        mod = import_module(*args, **kwargs)
    except ModuleNotFoundError as e:
        mod = None
        logger.name = logger_name
        logger.debug("Ignored: %s", str(e))
    return mod
