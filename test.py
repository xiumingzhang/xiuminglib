# For blaze on Google's infrastructure

from absl import app

from google3.experimental.users.xiuming.xiuminglib import xiuminglib as xm


def main(_):
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
