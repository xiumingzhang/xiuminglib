from os import remove
from os.path import abspath, dirname, exists
try:
    import bpy
except ModuleNotFoundError:
    # For building the doc
    pass

import xiuminglib as xm

logger, thisfile = xm.config.create_logger(abspath(__file__))


def save_blend(outpath=None, delete_overwritten=False):
    """Saves current scene to a .blend file.

    Args:
        outpath (str, optional): Path to save the scene to, e.g., ``'~/foo.blend'``. ``None`` means saving to the current file.
        delete_overwritten (bool, optional): Whether to delete or keep as .blend1 the same-name file.

    Writes:
        - A .blend file.
    """
    logger_name = thisfile + '->save_blend()'

    if outpath is None:
        outpath = ''
    else:
        xm.general.makedirs(dirname(outpath))
        if exists(outpath) and delete_overwritten:
            remove(outpath)

    try:
        # bpy.ops.file.autopack_toggle()
        bpy.ops.file.pack_all()
    except RuntimeError:
        logger.name = logger_name
        logger.error("Failed to pack some files")

    bpy.ops.wm.save_as_mainfile(filepath=outpath)

    logger.name = logger_name
    if outpath == '':
        msg = "Saved to the original .blend file"
    else:
        msg = "Saved to %s" % outpath
    logger.info(msg)


def open_blend(inpath):
    """Opens a .blend file.

    Args:
        inpath (str): E.g., ``'~/foo.blend'``.
    """
    bpy.ops.wm.open_mainfile(filepath=inpath)
