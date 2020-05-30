# For blaze on Google's infrastructure

import numpy as np

from absl import app

try:
    from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm
except ModuleNotFoundError:
    import xiuminglib as xm


def main(_):
    dtype = 'uint16'
    n_ch = 3
    ims = 256
    ssim = xm.metric.SSIM(dtype)
    dtype_max = np.iinfo(dtype).max
    im1 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    im2 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    print(ssim.compute(im1, im2))

    return
    dtype = 'uint8'
    n_ch = 3
    ims = 256
    psnr = xm.metric.PSNR(dtype)
    dtype_max = np.iinfo(dtype).max
    im1 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    im2 = (np.random.rand(ims, ims, n_ch) * dtype_max).astype(dtype)
    print(psnr.compute(im1, im2))

    return

    im_linear = np.random.rand(256, 256, 3)
    im_srgb = xm.img.linear2srgb(im_linear)
    im_srgb_linear = xm.img.srgb2linear(im_srgb)
    print(np.abs(im_linear - im_srgb_linear).max())

    return

    imgs = [
        '/usr/local/google/home/xiuming/Desktop/normalize_energy/302_20190801_103352/fullylit_scaled_cam.png',
        '/usr/local/google/home/xiuming/Desktop/normalize_energy/302_20190801_103352/olat_cam.png',
    ]
    labels = [
        "Scaled Fully Lit",
        "OLAT",
    ]
    xm.vis.video.make_apng(
        imgs, labels=labels, outpath='/usr/local/google/home/xiuming/Desktop/test.apng',
        font_ttf='/cns/ok-d/home/gcam-eng/gcam/interns/xiuming/relight/data/fonts/open-sans/OpenSans-Regular.ttf')


if __name__ == '__main__':
    app.run(main)
