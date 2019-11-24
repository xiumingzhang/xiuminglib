from os.path import abspath, join
from time import time

from ..config import create_logger
logger, thisfile = create_logger(abspath(__file__))

from .. import const
from ..imprt import preset_import


def make_apng(imgs, fps=24, outpath=None,
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
    logger_name = thisfile + '->make_apng()'

    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_apng.apng')

    img_paths = []
    for img in imgs:
        if isinstance(img, str):
            img_paths.append(img)
        else:
            tmpdir = join(const.Dir.tmp, 'make_apng_workspace')


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
