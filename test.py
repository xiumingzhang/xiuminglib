# For blaze on Google's infrastructure

import numpy as np

from absl import app

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


def main(_):
    im1 = np.random.rand(256, 256, 3)
    im2 = np.random.rand(256, 256, 3)
    ssim = xm.metric.SSIM(1)
    score = ssim.compute(im1, im2)

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
