from os.path import abspath, join, dirname
from time import time
import numpy as np

from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .. import const
from ..os import makedirs, call
from ..imprt import preset_import


def make_apng(imgs, interval=1, outpath=None):
    """Writes a list of images into an animated PNG.

    Args:
        imgs (list(numpy.ndarray or str)): An image is either a path or an
            array (mixing ok). If array, should be of type ``uint`` and of
            shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        interval (float, optional): Flipping interval in seconds.
        outpath (str, optional): Where to write the output to (a .apng file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_apng.apng')``.

    Raises:
        TypeError: If any input image is neither a string nor an array.
        ValueError: If any input image is neither 2D (grayscale) nor 3D (color).

    Writes
        - An animated PNG of the images.
    """
    logger_name = thisfile + '->make_apng()'

    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_apng.apng')
    if not outpath.endswith('.apng'):
        outpath += '.apng'
    makedirs(dirname(outpath))

    # If some images are not provided as paths
    if not all(isinstance(x, str) for x in imgs):
        cv2 = preset_import('cv2')
        tmpdir = join(const.Dir.tmp, 'make_apng_tmp')
        makedirs(tmpdir)

    img_paths = []
    for img in imgs:
        if isinstance(img, str):
            img_paths.append(img)
        elif isinstance(img, np.ndarray):
            # Need to write to disk, because ffmpeg takes paths
            assert np.issubdtype(img.dtype, np.unsignedinteger), \
                "If image is provided as an array, it has to be `uint`"
            if img.ndim == 3:
                img = img[:, :, ::-1] # BGR
            elif img.ndim == 2:
                pass
            else:
                raise ValueError(img.ndim)
            tmpf = join(tmpdir, '%f.png' % time())
            cv2.imwrite(tmpf, img)
            img_paths.append(tmpf)
        else:
            raise TypeError(type(img))

    cmd = 'ffmpeg -r %f ' % interval
    cmd += '-i concat:"'
    for img_path in img_paths[:2]:
        cmd += img_path + '|'
    cmd = cmd[:-1] + '"'
    cmd += ' -plays 0 ' # loops infinitely
    cmd += outpath
    cmd += ' -y' # overwrites

    call(cmd)

    logger.name = logger_name
    logger.info("Images written as an animated PNG to:\n%s", outpath)


def make_video(imgs, fps=24, outpath=None,
               matplotlib=True, dpi=96, bitrate=7200):
    """Writes a list of images into a grayscale or color video.

    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        matplotlib (bool, optional): Whether to use ``matplotlib``.
            If ``False``, use ``cv2``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate when using ``matplotlib``.

    Writes
        - A video of the images.
    """
    logger_name = thisfile + '->make_video()'

    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_video.mp4')
    makedirs(dirname(outpath))

    h, w = imgs[0].shape[:2]

    if matplotlib:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)

    else:
        cv2 = preset_import('cv2')

        # fourcc = cv2.VideoWriter_fourcc(*'X264') # .avi
        fourcc = 0x00000021 # .mp4
        vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

        for frame in imgs:
            assert frame.shape[:2] == (h, w), \
                "All frames must have the same shape"
            if frame.ndim == 3:
                frame = frame[:, :, ::-1] # cv2 uses BGR
            vw.write(frame)

        vw.release()

    logger.name = logger_name
    logger.info("Images written as a video to:\n%s", outpath)
