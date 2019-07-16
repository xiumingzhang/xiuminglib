from os.path import abspath
from importlib import import_module

from .config import create_logger
logger, thisfile = create_logger(abspath(__file__))


def preset_import(name):
    """A unified importer for both regular and ``google3`` modules, according
    to specified presets/profiles (e.g., ignoring ``ModuleNotFoundError``).
    """
    if name == 'cv2':
        # BUILD dep:
        # "//third_party/py/cvx2",
        try:
            from cvx2 import latest as mod
        except ModuleNotFoundError:
            mod = import_module_404ok('cv2')
        # TODO: Below is cleaner, but doesn't work
        # mod = import_module_404ok('cvx2.latest')
        # if mod is None:
        #    mod = import_module_404ok('cv2')
        return mod

    if name == 'gfile':
        # BUILD deps:
        # "//pyglib:gfile",
        # "//file/colossus/cns",
        mod = import_module_404ok('google3.pyglib.gfile')
        return mod

    if name in ('bpy', 'bmesh', 'OpenEXR', 'Imath'):
        # BUILD deps:
        # "//third_party/py/Imath",
        # "//third_party/py/OpenEXR",
        mod = import_module_404ok(name)
        return mod

    if name in ('Vector', 'Matrix', 'Quaternion'):
        mod = import_module_404ok('mathutils')
        cls = get_class(mod, name)
        return cls

    if name == 'BVHTree':
        mod = import_module_404ok('mathutils.bvhtree')
        cls = get_class(mod, name)
        return cls

    raise NotImplementedError(name)


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
        logger.warning("Ignored: %s", str(e))
    return mod


def get_class(mod, clsname):
    if mod is None:
        return None
    return getattr(mod, clsname)
