from os import remove
from os.path import abspath, dirname, exists
try:
    import bpy
except ModuleNotFoundError:
    # For building the doc
    pass

from xiuminglib import general as xg

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def save_blend(outpath, delete_overwritten=False):
    """Saves current scene to a .blend file.

    Args:
        outpath (str): Path to save scene to, e.g., ``'~/foo.blend'``.
        delete_overwritten (bool, optional): Whether to delete or keep as .blend1 the same-name file.
    """
    logger_name = thisfile + '->save_blend()'

    outdir = dirname(outpath)
    xg.makedirs(outdir)
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
    logger.info("Saved to %s", outpath)


def open_blend(inpath):
    """Opens a .blend file.

    Args:
        inpath (str): E.g., ``'~/foo.blend'``.
    """
    bpy.ops.wm.open_mainfile(filepath=inpath)
