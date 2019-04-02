from os import makedirs, remove
from os.path import abspath, dirname, exists
try:
    import bpy
except ModuleNotFoundError:
    # For building the doc
    pass

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))


def save_blend(outpath, delete_overwritten=False):
    """
    Save current scene to .blend file

    Args:
        outpath: Path to save scene to, e.g., '~/foo.blend'
            String
        delete_overwritten: Whether to delete or keep as .blend1 the same-name file
            Boolean
            Optional; defaults to False
    """
    logger_name = thisfile + '->save_blend()'

    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)
    elif exists(outpath) and delete_overwritten:
        remove(outpath)

    try:
        bpy.ops.file.autopack_toggle()
    except RuntimeError:
        logger.name = logger_name
        logger.error("Failed to pack some files")

    bpy.ops.wm.save_as_mainfile(filepath=outpath)

    logger.name = logger_name
    logger.info("Saved to %s", outpath)
