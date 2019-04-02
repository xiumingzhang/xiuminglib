import numpy as np


class PerspCamera(object):
    def __init__(self, f=50., im_res=(256, 256), loc=(0, 0, 0), lookat=(0, 0, 0), up=(0, 1, 0)):
        """
        Initialize a perspective camera in 35mm format
        Note:
            (0) Sensor width of 35mm format is actually 36mm
            (1) Assuming unit pixel aspect ratio (i.e., f_x = f_y) and
                no skewing between sensor plane and optical axis
            (2) Active sensor size may be smaller than sensor_size, depending on im_res
            (3) aov is a hardware property, having nothing to do with im_res

        Args:
            f: 35mm format-equivalent focal length in millimeters
                Float
                Optional; defaults to 50
            im_res: Image height and width in pixels
                Array_like of two positive integers
                Optional; defaults to (256, 256)
            loc: Camera location (in object space)
                Array_like of three floats
                Optional; defaults to (0, 0, 0)
            lookat: Where the camera points to (in object space)
                Array_like of three floats
                Optional; defaults to object center
            up: Vector (in object space) that, when projected, points upward in image
                Array_like of three floats
                Optional; defaults to (0, 1, 0)
        """
        self.f_mm = f
        self.im_h, self.im_w = im_res
        self.loc = np.array(loc)
        self.lookat = np.array(lookat)
        self.up = np.array(up)

    @property
    def sensor_w(self):
        return 36 # mm

    @property
    def sensor_h(self):
        return 24 # mm

    @property
    def aov(self):
        """
        Vertical and horizontal angles of view in degrees
        Tuple of two floats
        """
        alpha_v = 2 * np.arctan(self.sensor_h / (2 * self.f_mm))
        alpha_h = 2 * np.arctan(self.sensor_w / (2 * self.f_mm))
        return (alpha_v / np.pi * 180, alpha_h / np.pi * 180)

    @property
    def _mm_per_pix(self):
        return min(self.sensor_h / self.im_h, self.sensor_w / self.im_w)

    @property
    def f_pix(self):
        """
        Focal length in pixels
        Float
        """
        return self.f_mm / self._mm_per_pix

    @property
    def int_mat(self):
        """
        Intrinsics matrix
        (3, 3)-numpy array of floats
        """
        return np.array([
            [self.f_pix, 0, self.im_w / 2],
            [0, self.f_pix, self.im_h / 2],
            [0, 0, 1],
        ])

    @property
    def ext_mat(self):
        """
        Extrinsics matrix, i.e., rotation and translation that transform
            a point from object space to camera space
        (3, 4)-numpy array of floats
        """
        # Two coordinate systems involved:
        #   1. Object space: "obj"
        #   2. Desired computer vision camera coordinates: "cv"
        #        - x is horizontal, pointing right (to align with pixel coordinates)
        #        - y is vertical, pointing down
        #        - right-handed: positive z is the look-at direction

        # cv axes expressed in obj space
        cvz_obj = self.lookat - self.loc
        cvx_obj = np.cross(cvz_obj, self.up)
        cvy_obj = np.cross(cvz_obj, cvx_obj)
        # Normalize
        cvz_obj = cvz_obj / np.linalg.norm(cvz_obj)
        cvx_obj = cvx_obj / np.linalg.norm(cvx_obj)
        cvy_obj = cvy_obj / np.linalg.norm(cvy_obj)

        # Compute rotation from obj to cv: R
        # R(1, 0, 0)^T = cvx_obj gives first column of R
        # R(0, 1, 0)^T = cvy_obj gives second column of R
        # R(0, 0, 1)^T = cvz_obj gives third column of R
        rot_obj2cv = np.vstack((cvx_obj, cvy_obj, cvz_obj)).T

        # Extrinsics
        return rot_obj2cv.dot(
            np.array([
                [1, 0, 0, -self.loc[0]],
                [0, 1, 0, -self.loc[1]],
                [0, 0, 1, -self.loc[2]],
            ])
        )

    @property
    def proj_mat(self):
        """
        Projection matrix from intrinsics and extrinsics
        (3, 4)-numpy array of floats
        """
        return self.int_mat.dot(self.ext_mat)

    def set_from_mitsuba(self, xml_path):
        """
        Set camera with a Mitsuba XML file

        Args:
            xml_path: XML file path
                String
        """
        from xml.etree.ElementTree import parse

        tree = parse(xml_path)

        # Focal length
        f_tag = tree.find('./sensor/string[@name="focalLength"]')
        if f_tag is None:
            self.f_mm = 50. # Mitsuba default
        else:
            f_str = f_tag.attrib['value']
            if f_str[-2:] == 'mm':
                self.f_mm = float(f_str[:-2])
            else:
                raise NotImplementedError(f_str)

        # Extrinsics
        cam_transform = tree.find('./sensor/transform/lookAt').attrib
        self.loc = np.fromstring(cam_transform['origin'], sep=',')
        self.lookat = np.fromstring(cam_transform['target'], sep=',')
        self.up = np.fromstring(cam_transform['up'], sep=',')

        # Resolution
        self.im_h = int(tree.find('./sensor/film/integer[@name="height"]').attrib['value'])
        self.im_w = int(tree.find('./sensor/film/integer[@name="width"]').attrib['value'])

    def proj(self, pts, space='object'):
        """
        Project 3D points

        Args:
            pts: 3D points
                Float array_like of shape (n, 3), (3, n), or (3,)
            space: In which space these points are specified
                'object' or 'camera'
                Optional; defaults to 'object'

        Returns:
            vhs: Vertical and horizontal coordinates of the projections
                Float array_like of shape (n, 2) or (2,)
                +-----------> dim1
                |
                |
                |
                v dim0
        """
        pts = np.array(pts)
        if pts.shape == (3,):
            pts = pts.reshape((3, 1))
        elif pts.shape[1] == 3:
            pts = pts.T
        assert space in ('object', 'camera'), "Unrecognized space"

        # 3 x N
        n_pts = pts.shape[1]
        pts_homo = np.vstack((pts, np.ones((1, n_pts))))
        # 4 x N

        if space == 'object':
            proj_mat = self.proj_mat
        else:
            ext_mat = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj_mat = self.int_mat.dot(ext_mat)

        # Project
        hvs_homo = proj_mat.dot(pts_homo)
        # 3 x N: dim0 is horizontal, and dim1 is vertical

        hs_homo = hvs_homo[0, :]
        vs_homo = hvs_homo[1, :]
        ws = hvs_homo[2, :]
        hs = np.divide(hs_homo, ws)
        vs = np.divide(vs_homo, ws)

        vhs = np.vstack((vs, hs)).T
        if vhs.shape[0] == 1:
            # Single point
            vhs = vhs[0, :]
        return vhs

    def backproj(self, depth, fg_mask=None, depth_type='plane', space='object'):
        """
        Backproject depth map to 3D points

        Args:
            depth: Depth map
                2D numpy array of floats
            fg_mask: Backproject only pixels falling inside this foreground mask
                2D numpy array of logicals
                Optional; defaults to all Trues
            depth_type: Plane or ray depth
                String
                Optional; defaults to 'plane'
            space: In which space the backprojected points are specified
                'object' or 'camera'
                Optional; defaults to 'object'

        Returns:
            pts: 3D points
                N-by-3 numpy array of floats
        """
        if fg_mask is None:
            fg_mask = np.ones(depth.shape, dtype=bool)
        assert depth_type in ('ray', 'plane'), "Unrecognized depth type"
        assert space in ('object', 'camera'), "Unrecognized space"

        v_is, h_is = np.where(fg_mask)
        hs = h_is + 0.5
        vs = v_is + 0.5
        h_c = (depth.shape[1] - 1) / 2
        v_c = (depth.shape[0] - 1) / 2
        zs = depth[fg_mask]

        if depth_type == 'ray':
            d2 = np.power(vs - v_c, 2) + np.power(hs - h_c, 2)
            # Similar triangles
            zs_plane = np.multiply(zs, self.f_pix / np.sqrt(self.f_pix ** 2 + d2))
            zs = zs_plane

        # Backproject to camera space
        xs = np.multiply(zs, hs - h_c) / self.f_pix
        ys = np.multiply(zs, vs - v_c) / self.f_pix
        pts = np.vstack((xs, ys, zs))

        if space == 'camera':
            return pts.T

        # Need to further transform to object space
        rot_mat = self.ext_mat[:, :3] # happens first in projection
        trans_vec = self.ext_mat[:, 3].reshape(-1, 1) # happens second in projection
        n_pts = pts.shape[1]
        pts_obj = np.linalg.inv(rot_mat).dot(pts - np.tile(trans_vec, (1, n_pts)))

        return pts_obj.T
