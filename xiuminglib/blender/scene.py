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

    if outpath is not None:
        # "Save as" scenario: delete and then save
        xm.general.makedirs(dirname(outpath))
        if exists(outpath) and delete_overwritten:
            remove(outpath)

    try:
        # bpy.ops.file.autopack_toggle()
        bpy.ops.file.pack_all()
    except RuntimeError:
        logger.name = logger_name
        logger.error("Failed to pack some files")

    if outpath is None:
        # "Save" scenario: save and then delete
        bpy.ops.wm.save_as_mainfile()
        outpath = bpy.context.blend_data.filepath
        bakpath = outpath + '1'
        if exists(bakpath) and delete_overwritten:
            remove(bakpath)
    else:
        bpy.ops.wm.save_as_mainfile(filepath=outpath)

    logger.name = logger_name
    logger.info("Saved to %s", outpath)


def open_blend(inpath):
    """Opens a .blend file.

    Args:
        inpath (str): E.g., ``'~/foo.blend'``.
    """
    bpy.ops.wm.open_mainfile(filepath=inpath)
