from importlib import import_module

from .decor import mod404ok


def preset_import(module_name):
    """A unified importer for both regular and ``google3`` modules, according
    to specified presets/profiles (e.g., ignoring ``ModuleNotFoundError``).
    """
    import_404ok = mod404ok(import_module)

    if module_name == 'cv2':
        # Try first assuming Blaze
        # "//third_party/py/cvx2"
        mod = import_404ok('cvx2')
        if mod is None:
            mod = import_404ok('cv2')

    elif module_name == 'gfile':
        # "//pyglib:gfile"
        # "//file/colossus/cns"
        mod = import_404ok('gfile', package='google3.pyglib')

    elif module_name in ('bpy', 'bmesh', 'OpenEXR', 'Imath'):
        mod = import_404ok(module_name)

    elif module_name in ('Vector', 'Matrix', 'Quaternion'):
        mod = import_404ok(module_name, package='mathutils')

    elif module_name == 'BVHTree':
        mod = import_404ok(module_name, package='mathutils.bvhtree')

    else:
        raise NotImplementedError(module_name)

    return mod
