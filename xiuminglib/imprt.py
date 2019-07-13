def import_from_google3(module_name):
    """Imports a module from ``google3``."""
    if module_name == 'cv2':
        try:
            # "//third_party/py/cvx2"
            import cvx2 as cv2
            # from google3.third_party.OpenCVX import cvx2 as cv2
            # also works
        except ModuleNotFoundError:
            import cv2
        return cv2

    if module_name == 'gfile':
        try: # using Blaze
            # "//pyglib:gfile"
            from google3.pyglib import gfile
        except ModuleNotFoundError: # not using
            # It's OK, as we can use the fileutil CLI
            gfile = None
        return gfile

    raise NotImplementedError(module_name)
