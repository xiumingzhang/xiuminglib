"""
Tracker classes

Xiuming Zhang, MIT CSAIL
November 2017
"""

from os.path import join
import numpy as np
import cv2
from xiuminglib import visualization as xvis


class LucasKanadeTracker:
    def __init__(self, frames, pts, backtrack_thres=1, lk_params=None):
        """
        Args:
            frames: Frame images in order
                List of h-by-w or h-by-w-by-3 numpy arrays
                Color images will be converted to grayscale
            pts: Points to track in the first frame
                Array_like of shape (n, 2)
                    +------------>
                    |       pts[:, 1]
                    |
                    |
                    v pts[:, 0]
            backtrack_thres: Largest pixel deviation in x or y direction of
                a successful backtrack
                Float
                Optional; defaults to 1
            lk_params: Keyword parameters for calcOpticalFlowPyrLK()
                Dictionary of parameter name-value pairs
                Optional

        Result attrs:
            tracks: Positions of tracks from the i-th to (i+1)-th frame
                List of n-by-2 numpy arrays
                    +------------>
                    |       tracks[:, 1]
                    |
                    |
                    v tracks[:, 0]
            can_backtrack: Whether each track can be back-tracked to the previous frame
                List of Boolean numpy arrays of length n
            is_lost: Whether each track is lost in this frame
                List of Boolean numpy arrays of length n
        """
        frames_gs = []
        for img in frames:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames_gs.append(img)
        self.frames = frames_gs

        self.pts = np.array(pts)

        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 12,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        if lk_params is not None:
            # Overwrite with whatever is user-provided
            for key, val in lk_params.items():
                self.lk_params[key] = val

        self.backtrack_thres = backtrack_thres
        self.tracks = []
        self.can_backtrack = []
        self.is_lost = []

    def run(self, constrain=None):
        """
        Args:
            constrain: Function applied to tracks before being fed to the next round
                function that takes in an n-by-2 numpy array as well as the current workspace
                (as a dictionary) and returns another n-by-2 numpy array
                Optional; defaults to None
        """
        for fi in range(0, len(self.frames) - 1):
            f0, f1 = self.frames[fi], self.frames[fi + 1]

            if fi == 0:
                p0 = self._my2klt(self.pts)

            # Track with forward flow
            p1, not_lost, err = cv2.calcOpticalFlowPyrLK(f0, f1, p0, None, **self.lk_params)
            is_lost = (1 - not_lost.ravel()).astype(bool)
            err = err.ravel()

            # Check quality by back-tracking
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(f1, f0, p1, None, **self.lk_params)
            can_backtrack = abs(p0 - p0r).reshape(-1, 2).max(-1) < self.backtrack_thres

            # Continue tracking these points or impose some constraints
            if constrain is None:
                p0 = p1
            else:
                pts = self._klt2my(p1)
                pts = constrain(pts, locals())
                p0 = self._my2klt(pts)

            self.tracks.append(self._klt2my(p0))
            self.can_backtrack.append(can_backtrack)
            self.is_lost.append(is_lost)

    def vis(self, out_dir, marker_bgr=(0, 0, 255)):
        for fi in range(0, len(self.frames) - 1):
            im = self.frames[fi + 1]
            pts = self.tracks[fi]
            xvis.scatter_on_image(im, pts, size=6, bgr=marker_bgr,
                                  outpath=join(out_dir, '%04d.png' % (fi + 1)))

    @staticmethod
    def _my2klt(pts):
        """
        Reshaping
            +------------>
            |       pts[:, 1]
            |
            |
            v pts[:, 0]
        to
            +------------>
            |       pts[:, 0, 0]
            |
            |
            v pts[:, 0, 1]
        """
        return np.expand_dims(
            np.vstack((pts[:, 1],
                       pts[:, 0])).T, 1).astype(np.float32)

    @staticmethod
    def _klt2my(pts):
        """
        Inverse of _my2klt()
        """
        return pts.reshape(-1, 2)[:, ::-1]
