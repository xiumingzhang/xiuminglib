from os.path import join, dirname
import numpy as np

from ..log import get_logger
logger = get_logger()

from .. import const
from ..os import makedirs
from ..imprt import preset_import


def make_video(
        imgs, fps=24, outpath=None, method='matplotlib', dpi=96, bitrate=-1):
    """Writes a list of images into a grayscale or color video.

    Args:
        imgs (list(numpy.ndarray)): Each image should be of type ``uint8`` or
            ``uint16`` and of shape H-by-W (grayscale) or H-by-W-by-3 (RGB).
        fps (int, optional): Frame rate.
        outpath (str, optional): Where to write the video to (a .mp4 file).
            ``None`` means
            ``os.path.join(const.Dir.tmp, 'make_video.mp4')``.
        method (str, optional): Method to use: ``'matplotlib'``, ``'opencv'``,
            ``'video_api'``.
        dpi (int, optional): Dots per inch when using ``matplotlib``.
        bitrate (int, optional): Bit rate in kilobits per second when using
            ``matplotlib``; reasonable values include 7200.

    Writes
        - A video of the images.
    """
    if outpath is None:
        outpath = join(const.Dir.tmp, 'make_video.mp4')
    makedirs(dirname(outpath))

    assert imgs, "Frame list is empty"
    for frame in imgs:
        assert np.issubdtype(frame.dtype, np.unsignedinteger), \
            "Image type must be unsigned integer"

    h, w = imgs[0].shape[:2]
    for frame in imgs[1:]:
        assert frame.shape[:2] == (h, w), \
            "All frames must have the same shape"

    if method == 'matplotlib':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import animation

        w_in, h_in = w / dpi, h / dpi
        fig = plt.figure(figsize=(w_in, h_in))
        Writer = animation.writers['ffmpeg'] # may require you to specify path
        writer = Writer(fps=fps, bitrate=bitrate)

        def img_plt(arr):
            img_plt_ = plt.imshow(arr)
            ax = plt.gca()
            ax.set_position([0, 0, 1, 1])
            ax.set_axis_off()
            return img_plt_

        anim = animation.ArtistAnimation(fig, [(img_plt(x),) for x in imgs])
        anim.save(outpath, writer=writer)
        # If obscure error like "ValueError: Invalid file object: <_io.Buff..."
        # occurs, consider upgrading matplotlib so that it prints out the real,
        # underlying ffmpeg error

        plt.close('all')

    elif method == 'opencv':
        cv2 = preset_import('cv2', assert_success=True)

        # TODO: debug codecs (see http://www.fourcc.org/codecs.php)
        if outpath.endswith('.mp4'):
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'X264')
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # fourcc = 0x00000021
        elif outpath.endswith('.avi'):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            raise NotImplementedError("Video type of\n\t%s" % outpath)

        vw = cv2.VideoWriter(outpath, fourcc, fps, (w, h))

        for frame in imgs:
            if frame.ndim == 3:
                frame = frame[:, :, ::-1] # cv2 uses BGR
            vw.write(frame)

        vw.release()

    elif method == 'video_api':
        video_api = preset_import('video_api', assert_success=True)

        assert outpath.endswith('.webm'), "`video_api` requires .webm"

        with video_api.write(outpath, fps=fps) as h:
            for frame in imgs:
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                #frame = frame.astype(np.ubyte)
                h.add_frame(frame)

    else:
        raise ValueError(method)

    logger.info("Images written as a video to:\n%s", outpath)
