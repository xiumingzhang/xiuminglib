from os import remove, rename
from os.path import abspath, dirname, basename
from time import time
import numpy as np
try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix, Quaternion
    from mathutils.bvhtree import BVHTree
except ModuleNotFoundError:
    # For building the doc
    pass

from xiuminglib import config
logger, thisfile = config.create_logger(abspath(__file__))

from .object import get_bmesh


def add_camera(xyz=(0, 0, 0),
               rot_vec_rad=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=100):
    """Adds a camera to  the current scene.

    Args:
        xyz (tuple, optional): Location. Defaults to ``(0, 0, 0)``.
        rot_vec_rad (tuple, optional): Rotations in radians around x, y and z.
            Defaults to ``(0, 0, 0)``.
        name (str, optional): Camera object name.
        proj_model (str, optional): Camera projection model. Must be ``'PERSP'``,
            ``'ORTHO'``, or ``'PANO'``. Defaults to ``'PERSP'``.
        f (float, optional): Focal length in mm. Defaults to 35.
        sensor_fit (str, optional): Sensor fit. Must be ``'HORIZONTAL'`` or ``'VERTICAL'``.
            See also :func:`get_camera_matrix`. Defaults to ``'HORIZONTAL'``.
        sensor_width (float, optional): Sensor width in mm. Defaults to 32.
        sensor_height (float, optional): Sensor height in mm. Defaults to 18.
        clip_start (float, optional): Near clipping distance. Defaults to 0.1.
        clip_end (float, optional): Far clipping distance. Defaults to 100.

    Returns:
        bpy_types.Object: Camera added.
    """
    logger_name = thisfile + '->add_camera()'

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end

    logger.name = logger_name
    logger.info("Camera '%s' added", cam.name)

    return cam


def easyset(cam,
            xyz=None,
            rot_vec_rad=None,
            name=None,
            proj_model=None,
            f=None,
            sensor_fit=None,
            sensor_width=None,
            sensor_height=None):
    """Sets camera parameters more easily.

    See :func:`add_camera` for arguments. ``None`` will result in no change.
    """
    if name is not None:
        cam.name = name

    if xyz is not None:
        cam.location = xyz

    if rot_vec_rad is not None:
        cam.rotation_euler = rot_vec_rad

    if proj_model is not None:
        cam.data.type = proj_model

    if f is not None:
        cam.data.lens = f

    if sensor_fit is not None:
        cam.data.sensor_fit = sensor_fit

    if sensor_width is not None:
        cam.data.sensor_width = sensor_width

    if sensor_height is not None:
        cam.data.sensor_height = sensor_height


def point_camera_to(cam, xyz_target, up=(0, 0, 1)):
    """Points camera to target.

    Args:
        cam (bpy_types.Object): Camera object.
        xyz_target (array_like): Target point in world coordinates.
        up (array_like, optional): World vector that, when projected,
            points up in the image plane.
    """
    logger_name = thisfile + '->point_camera_to()'

    up = Vector(up)
    xyz_target = Vector(xyz_target)

    direction = xyz_target - cam.location

    # Rotate camera with quaternion so that `track` aligns with `direction`, and
    # world +z, when projected, aligns with camera +y (i.e., points up in image plane)
    track = '-Z'
    rot_quat = direction.to_track_quat(track, 'Y')
    cam.rotation_euler = (0, 0, 0)
    cam.rotation_euler.rotate(rot_quat)

    # Further rotate camera so that world `up`, when projected, points up on image plane
    # We know right now world +z, when projected, points up, so we just need to rotate
    # the camera around the lookat direction by an angle
    cam_mat, _, _ = get_camera_matrix(cam)
    up_proj = cam_mat * up.to_4d()
    orig_proj = cam_mat * Vector((0, 0, 0)).to_4d()
    try:
        up_proj = Vector((up_proj[0] / up_proj[2], up_proj[1] / up_proj[2])) - \
            Vector((orig_proj[0] / orig_proj[2], orig_proj[1] / orig_proj[2]))
    except ZeroDivisionError:
        logger.name = logger_name
        logger.error(
            ("w in homogeneous coordinates is 0; "
             "camera coincides with the point to project? "
             "So can't rotate camera to ensure up vector")
        )
        logger.info("Camera '%s' pointed to %s, but with no guarantee on up vector",
                    cam.name, tuple(xyz_target))
        return cam
    # +------->
    # |
    # |
    # v
    up_proj[1] = -up_proj[1]
    # ^
    # |
    # |
    # +------->
    a = Vector((0, 1)).angle_signed(up_proj) # clockwise is positive
    cam.rotation_euler.rotate(Quaternion(direction, a))

    logger.name = logger_name
    logger.info("Camera '%s' pointed to %s with world %s pointing up",
                cam.name, tuple(xyz_target), tuple(up))

    return cam


def intrinsics_compatible_with_scene(cam, eps=1e-6):
    r"""Checks if camera intrinsic parameters are comptible with the current scene.

    Intrinsic parameters include sensor size and pixel aspect ratio, and scene
    parameters refer to render resolutions and their scale. The entire sensor is
    assumed active.

    Args:
        cam (bpy_types.Object): Camera object
        eps (float, optional): :math:`\epsilon` for numerical comparison. Considered
            equal if :math:`\frac{|a - b|}{b} < \epsilon`.

    Returns:
        bool: Check result.
    """
    logger.name = thisfile + '->intrinsics_compatible_with_scene()'

    # Camera
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Do these parameters make sense together?
    mm_per_pix_horizontal = sensor_width_mm / (w * scale)
    mm_per_pix_vertical = sensor_height_mm / (h * scale)

    if abs(mm_per_pix_horizontal / mm_per_pix_vertical - pixel_aspect_ratio) \
            / pixel_aspect_ratio < eps:
        logger.info("OK")
        return True

    logger.error((
        "Render resolutions (w_pix = %d; h_pix = %d), active sensor size (w_mm = %f; "
        "h_mm = %f), and pixel aspect ratio (r = %f) don't make sense together. "
        "This could cause unexpected behaviors later. "
        "Consider running correct_sensor_height()"
    ), w, h, sensor_width_mm, sensor_height_mm, pixel_aspect_ratio)
    return False


def correct_sensor_height(cam):
    r"""To make render resolutions, sensor size, and pixel aspect ratio comptible.

    If render resolutions are :math:`(w_\text{pix}, h_\text{pix})`, sensor sizes
    are :math:`(w_\text{mm}, h_\text{mm})`, and pixel aspect ratio is :math:`r`,
    then :math:`h_\text{mm}\leftarrow\frac{h_\text{pix}}{w_\text{pix}r}w_\text{mm}`.

    Args:
        cam (bpy_types.Object): Camera.
    """
    logger_name = thisfile + '->correct_sensor_height()'

    # Camera
    sensor_width_mm = cam.data.sensor_width

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Change sensor height
    sensor_height_mm = sensor_width_mm * h / w / pixel_aspect_ratio
    cam.data.sensor_height = sensor_height_mm

    logger.name = logger_name
    logger.info("Sensor height changed to %f", sensor_height_mm)


def get_camera_matrix(cam, keep_disparity=False):
    r"""Gets camera matrix, intrinsics, and extrinsics from a camera.

    You can ask for a 4-by-4 projection that projects :math:`(x, y, z, 1)` to
    :math:`(u, v, 1, d)`, where :math:`d` is the disparity, reciprocal of depth.

    Args:
        cam (bpy_types.Object): Camera.
        keep_disparity (bool, optional): Whether matrices keep disparity or not.

    Raises:
        ValueError: If render settings and camera intrinsics mismatch. Run
            :func:`intrinsics_compatible_with_scene` for advice.

    Returns:
        tuple:
            - **cam_mat** (*mathutils.Matrix*) -- Camera matrix, product of intrinsics and
              extrinsics. 4-by-4 if ``keep_disparity``; else, 3-by-4.
            - **int_mat** (*mathutils.Matrix*) -- Camera intrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-3.
            - **ext_mat** (*mathutils.Matrix*) -- Camera extrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-4.
    """
    logger_name = thisfile + '->get_camera_matrix()'

    # Necessary scene update
    scene = bpy.context.scene
    scene.update()

    # Check if camera intrinsic parameters comptible with render settings
    if not intrinsics_compatible_with_scene(cam):
        raise ValueError(
            ("Render settings and camera intrinsic parameters mismatch. "
             "Such computed matrices will not make sense. Make them consistent first. "
             "See error message from 'intrinsics_compatible_with_scene()' above for advice")
        )

    # Intrinsics

    f_mm = cam.data.lens
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if cam.data.sensor_fit == 'VERTICAL':
        # h times pixel height must fit into sensor_height_mm
        # w / pixel_aspect_ratio times pixel width will then fit into sensor_width_mm
        s_v = h * scale / sensor_height_mm
        s_u = w * scale / pixel_aspect_ratio / sensor_width_mm
    else: # 'HORIZONTAL' or 'AUTO'
        # w times pixel width must fit into sensor_width_mm
        # h * pixel_aspect_ratio times pixel height will then fit into sensor_height_mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = w * scale / sensor_width_mm
        s_v = h * scale * pixel_aspect_ratio / sensor_height_mm

    skew = 0 # only use rectangular pixels

    if keep_disparity:
        # 4-by-4
        int_mat = Matrix((
            (s_u * f_mm, skew, w * scale / 2, 0),
            (0, s_v * f_mm, h * scale / 2, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1)))
    else:
        # 3-by-3
        int_mat = Matrix((
            (s_u * f_mm, skew, w * scale / 2),
            (0, s_v * f_mm, h * scale / 2),
            (0, 0, 1)))

    # Extrinsics

    # Three coordinate systems involved:
    #   1. World coordinates: "world"
    #   2. Blender camera coordinates: "cam"
    #        - x is horizontal
    #        - y is up
    #        - right-handed: negative z is look-at direction
    #   3. Desired computer vision camera coordinates: "cv"
    #        - x is horizontal
    #        - y is down (to align to the actual pixel coordinates)
    #        - right-handed: positive z is look-at direction

    rotmat_cam2cv = Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)))

    # matrix_world defines local-to-world transformation, i.e.,
    # where is local (x, y, z) in world coordinate system?
    t, rot_euler = cam.matrix_world.decompose()[0:2]

    # World to Blender camera
    rotmat_world2cam = rot_euler.to_matrix().transposed() # equivalent to inverse
    t_world2cam = rotmat_world2cam * -t

    # World to computer vision camera
    rotmat_world2cv = rotmat_cam2cv * rotmat_world2cam
    t_world2cv = rotmat_cam2cv * t_world2cam

    if keep_disparity:
        # 4-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],),
            (0, 0, 0, 1)))
    else:
        # 3-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],)))

    # Camera matrix
    cam_mat = int_mat * ext_mat

    logger.name = logger_name
    logger.info("Done computing camera matrix for '%s'", cam.name)
    logger.warning("    ... using w = %d; h = %d", w * scale, h * scale)

    return cam_mat, int_mat, ext_mat


def get_camera_zbuffer(cam, save_to=None, hide=None):
    """Gets :math:`z`-buffer of the camera.

    Values are :math:`z` components in camera-centered coordinate system, where

    - :math:`x` is horizontal;
    - :math:`y` is down (to align with the actual pixel coordinates);
    - right-handed: positive :math:`z` is look-at direction and means "in front of camera."

    Origin is camera center, not image plane (one focal length away from origin).

    Args:
        cam (bpy_types.Object): Camera.
        save_to (str, optional): Path to which the .exr :math:`z`-buffer will be saved.
            None means don't save.
        hide (str or list(str)): Names of objects to be hidden while rendering
            this camera's :math:`z`-buffer.

    Returns:
        numpy.ndarray: Camera :math:`z`-buffer.
    """
    import cv2

    logger_name = thisfile + '->get_camera_zbuffer()'

    # Validate and standardize error-prone inputs
    if hide is not None:
        if not isinstance(hide, list):
            # A single object
            hide = [hide]
        for element in hide:
            assert isinstance(element, str), \
                "`hide` should contain object names (i.e., strings), not objects themselves"

    if save_to is None:
        outpath = '/tmp/%s_zbuffer' % time()
    elif save_to.endswith('.exr'):
        outpath = save_to[:-4]

    # Duplicate scene to avoid touching the original scene
    bpy.ops.scene.new(type='LINK_OBJECTS')

    scene = bpy.context.scene
    scene.camera = cam
    scene.use_nodes = True
    node_tree = scene.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    # Set up nodes for z pass
    nodes.new('CompositorNodeRLayers')
    nodes.new('CompositorNodeOutputFile')
    node_tree.links.new(nodes['Render Layers'].outputs[2], nodes['File Output'].inputs[0])
    nodes['File Output'].format.file_format = 'OPEN_EXR'
    nodes['File Output'].format.color_mode = 'RGB'
    nodes['File Output'].format.color_depth = '32' # full float
    nodes['File Output'].base_path = dirname(outpath)
    nodes['File Output'].file_slots[0].path = basename(outpath)

    # Hide objects from z-buffer, if necessary
    if hide is not None:
        orig_hide_render = {} # for later restoration
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                orig_hide_render[obj.name] = obj.hide_render
                obj.hide_render = obj.name in hide

    # Render
    scene.cycles.samples = 1
    scene.render.filepath = '/tmp/%s_rgb.png' % time() # redirect RGB rendering to avoid overwritting
    bpy.ops.render.render(write_still=True)

    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Delete this new scene
    bpy.ops.scene.delete()

    # Restore objects' original render hide states, if necessary
    if hide is not None:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.hide_render = orig_hide_render[obj.name]

    # Load z-buffer as array
    exr_path = outpath + '%04d' % scene.frame_current + '.exr'
    im = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    assert (np.array_equal(im[:, :, 0], im[:, :, 1]) and np.array_equal(im[:, :, 0], im[:, :, 2])), \
        "BGR channels of the z-buffer should be all the same, but they are not"
    zbuffer = im[:, :, 0]

    # Delete or move the .exr as user wants
    if save_to is None:
        # User doesn't want it -- delete
        remove(exr_path)
    else:
        # User wants it -- rename
        rename(exr_path, outpath + '.exr')

    logger.name = logger_name
    logger.info("Got z-buffer of camera '%s'", cam.name)
    logger.warning("    ... using w = %d; h = %d", w * scale, h * scale)

    return zbuffer


def backproject_uv_to_3d(uvs, cam, obj_names=None, world_coords=False):
    """Backprojects 2D coordinates to 3D.

    Since a 2D point could be projected from any point on a 3D line,
    this function will return the 3D point at which this line (ray)
    intersects with an object for the first time.

    Args:
        uvs (array_like): UV coordinates of length 2 or shape N-by-2,
            in the following convention::
                (0, 0)
                +------------> (w, 0)
                |           u
                |
                |
                |
                v v (0, h)
        cam (bpy_types.Object): Camera.
        obj_names (str or list(str), optional): Names of objects of interest.
            ``None`` means all objects.
        world_coords (bool, optional): Whether to return world or local coordinates.

    Returns:
        tuple:
            - **xyzs** (*mathutils.Vector or list(mathutils.Vector)*) -- 3D local coordinates.
              Value being ``None`` means no intersections.
            - **intersect_objnames** (*str or list(str)*) -- Name(s) of object(s) responsible
              for intersections. ``None`` means no intersection.
    """
    logger_name = thisfile + '->backproject_uv_to_3d()'

    # Standardize inputs
    uvs = np.array(uvs).reshape(-1, 2)
    objs = bpy.data.objects
    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in objs if o.type == 'MESH']

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get 4-by-4 invertible camera matrix
    cam_mat, _, _ = get_camera_matrix(cam, keep_disparity=True)

    # Construct BVH trees for objects of interest
    trees = {}
    for obj_name in obj_names:
        obj = objs[obj_name]
        bm = get_bmesh(obj)
        trees[obj_name] = BVHTree.FromBMesh(bm)

    xyzs = [None] * uvs.shape[0]
    intersect_objnames = [None] * uvs.shape[0]

    for i in range(uvs.shape[0]):

        # Compute the infinitely far point on the line passing camera center and projecting to uv
        uv = uvs[i, :]
        uv1d = np.append(uv, [1, 0])
        xyzw = cam_mat.inverted() * Vector(uv1d) # w = 0; world

        # Ray start and direction in world coordinates
        ray_start_world = cam.location # origin in camera coordinates
        ray_dir_world = 1e10 * Vector(xyzw[:3]) - ray_start_world # boost it for robust matrix multiplications

        first_intersect = None
        first_intersect_objname = None
        dist_min = np.inf

        # Test intersections with each object of interest
        for obj_name, tree in trees.items():
            obj2world = objs[obj_name].matrix_world
            world2obj = obj2world.inverted()

            # Ray start and direction in local coordinates
            ray_start = world2obj * ray_start_world
            ray_dir = world2obj * ray_dir_world

            # Ray tracing
            loc, _, _, dist = tree.ray_cast(ray_start, ray_dir)

            # See if this intersection is closer to camera center
            if (dist is not None) and (dist < dist_min):
                if world_coords:
                    first_intersect = obj2world * loc
                else:
                    first_intersect = loc
                first_intersect_objname = obj_name

        xyzs[i] = first_intersect
        intersect_objnames[i] = first_intersect_objname

    logger.name = logger_name
    logger.info("Backprojection done with camera '%s'", cam.name)
    logger.warning("    ... using w = %d; h = %d", w * scale, h * scale)

    if uvs.shape[0] == 1:
        return xyzs[0], intersect_objnames[0]
    return xyzs, intersect_objnames


def get_visible_vertices(cam, obj, ignore_occlusion=False, perc_z_eps=1e-6, hide=None):
    r"""Gets vertices that are visible (projected within frame *and* unoccluded) from camera.

    Warning:
        Depth is considered the same within certain percentage, so the results may be inaccurate
        when object's own depth variation is small compared with its overall depth.

    Rasterized :math:`z`-buffer (instead of ray tracing) used for speed.
    Since :math:`z`-buffer may cover other objects, this function takes occlusion by other.
    objects into account, but you can opt to ignore the :math:`z`-buffer such that occluded
    vertices are also considered visible.

    Args:
        cam (bpy_types.Object): Camera.
        obj (bpy_types.Object): Object of interest.
        ignore_occlusion (bool, optional): Whether to ignore occlusion (including self-occlusion).
        perc_z_eps (float, optional): Threshold for percentage difference between the query :math:`z_q`
            and buffered :math:`z_b`. :math:`z_q` is considered equal to :math:`z_b` when
            :math:`\frac{|z_q - z_b|}{z_b} <` this. Useless if ``ignore_occlusion``.
        hide (str or list(str), optional): Names of objects to be hidden while rendering this camera's
            :math:`z`-buffer. Useless if ``ignore_occlusion``.

    Returns:
        list: Indices of vertices that are visible.
    """
    logger_name = thisfile + '->get_visible_vertices()'

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get camera matrix
    cam_mat, _, ext_mat = get_camera_matrix(cam)

    # Get z-buffer
    if not ignore_occlusion:
        zbuffer = get_camera_zbuffer(cam, hide=hide)

    # Get mesh data from object
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    visible_vert_ind = []
    # For each of its vertices
    for bv in bm.verts:

        # Check if its projection falls inside frame
        v_world = obj.matrix_world * bv.co # local to world
        uv = np.array(cam_mat * v_world) # project to 2D
        uv = uv[:-1] / uv[-1]
        if uv[0] >= 0 and uv[0] < w * scale and uv[1] >= 0 and uv[1] < h * scale:
            # Yes

            if ignore_occlusion:
                # Considered visible already
                visible_vert_ind.append(bv.index)
            else:
                # Proceed to check occlusion with z-buffer
                v_cv = ext_mat * v_world # world to camera to CV
                z = v_cv[-1]
                z_min = zbuffer[int(uv[1]), int(uv[0])]
                if (z - z_min) / z_min < perc_z_eps:
                    visible_vert_ind.append(bv.index)

    logger.name = logger_name
    logger.info("Visibility test done with camera '%s'", cam.name)
    logger.warning("    ... using w = %d; h = %d", w * scale, h * scale)

    return visible_vert_ind


def get_2d_bounding_box(obj, cam):
    """Gets a 2D bounding box of the object in the camera frame.

    This is different from projecting the 3D bounding box to 2D.

    Args:
        obj (bpy_types.Object): Object of interest.
        cam (bpy_types.Object): Camera.

    Returns:
        numpy.ndarray: 2D coordinates of the bounding box corners.
        Of shape 4-by-2. Corners are ordered counterclockwise, following::
            (0, 0)
            +------------> (w, 0)
            |           u
            |
            |
            |
            v v (0, h)
    """
    logger_name = thisfile + '->get_2d_bounding_box()'

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get camera matrix
    cam_mat, _, _ = get_camera_matrix(cam)

    # Project all vertices to 2D
    pts = np.vstack([v.co.to_4d() for v in obj.data.vertices]).T # 4-by-N
    world_mat = np.array(obj.matrix_world) # 4-by-4
    cam_mat = np.array(cam_mat) # 3-by-4
    uvw = cam_mat.dot(world_mat.dot(pts)) # 3-by-N
    pts_2d = np.divide(uvw[:2, :], np.tile(uvw[2, :], (2, 1))) # 2-by-N

    # Compute bounding box
    u_min, v_min = np.min(pts_2d, axis=1)
    u_max, v_max = np.max(pts_2d, axis=1)
    corners = np.vstack((
        np.array([u_min, v_min]),
        np.array([u_max, v_min]),
        np.array([u_max, v_max]),
        np.array([u_min, v_max])))

    logger.name = logger_name
    logger.info("Got 2D bounding box of '%s' in camera '%s'", obj.name, cam.name)
    logger.warning("    ... using w = %d; h = %d", w * scale, h * scale)

    return corners
