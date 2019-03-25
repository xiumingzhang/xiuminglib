"""
Utility functions for Blender renderings

Xiuming Zhang, MIT CSAIL
July 2017
"""

from os import makedirs
from os.path import abspath, dirname, exists, join
from shutil import move
from time import time
import bpy

import config
logger, thisfile = config.create_logger(abspath(__file__))


def set_cycles(w=None, h=None,
               n_samples=None, max_bounces=None, min_bounces=None,
               transp_bg=None,
               color_mode=None, color_depth=None):
    """
    Set up Cycles as rendering engine

    Args:
        w, h: Width, height of render in pixels
            Positive integer
            Optional; no change if not given
        n_samples: Number of samples
            Positive integer
            Optional; no change if not given
        max_bounces, min_bounces: Maximum, minimum number of light bounces
            Setting max_bounces to 0 for direct lighting only
            Natural number
            Optional; no change if not given
        transp_bg: Whether world background is transparent
            Boolean
            Optional; no change if not given
        color_mode: Color mode
            'BW', 'RGB' or 'RGBA'
            Optional; no change if not given
        color_depth: Color depth
            '8' or '16'
            Optional; no change if not given
    """
    logger_name = thisfile + '->set_cycles()'

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    cycles.use_progressive_refine = True
    if n_samples is not None:
        cycles.samples = n_samples
    if max_bounces is not None:
        cycles.max_bounces = max_bounces
    if min_bounces is not None:
        cycles.min_bounces = min_bounces
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 10
    cycles.glossy_bounces = 4
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 8
    cycles.transparent_max_bounces = 64

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = 5
    cycles.sample_clamp_indirect = 5

    # Ensure there's no background light emission
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    # If world background is transparent with premultiplied alpha
    if transp_bg is not None:
        cycles.film_transparent = transp_bg

    # # Use GPU
    # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
    # bpy.context.user_preferences.system.compute_device = 'CUDA_' + str(randint(0, 3))
    # scene.cycles.device = 'GPU'

    scene.render.tile_x = 16 # 256 optimal for GPU
    scene.render.tile_y = 16 # 256 optimal for GPU
    if w is not None:
        scene.render.resolution_x = w
    if h is not None:
        scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    if color_mode is not None:
        scene.render.image_settings.color_mode = color_mode
    if color_depth is not None:
        scene.render.image_settings.color_depth = color_depth

    logger.name = logger_name
    logger.info("Cycles set up as rendering engine")


def easyset(w=None, h=None,
            n_samples=None,
            ao=None,
            color_mode=None,
            file_format=None,
            color_depth=None):
    """
    Set some of the scene attributes more easily

    Args:
        w, h: Width, height of render in pixels
            Integer
            Optional; no change if not given
        n_samples: Number of samples
            Integer
            Optional; no change if not given
        ao: Ambient occlusion
            Boolean
            Optional; no change if not given
        color_mode: Color mode of rendering
            'BW', 'RGB', or 'RGBA'
            Optional; no change if not given
        file_format: File format of the render
            'PNG', 'OPEN_EXR', etc.
            Optional; no change if not given
        color_depth: Color depth of rendering
            '8' or '16' for .png; '16' or '32' for .exr
            Optional; no change if not given
    """
    scene = bpy.context.scene

    scene.render.resolution_percentage = 100

    if w is not None:
        scene.render.resolution_x = w

    if h is not None:
        scene.render.resolution_y = h

    # Number of samples
    if n_samples is not None and scene.render.engine == 'CYCLES':
        scene.cycles.samples = n_samples

    # Ambient occlusion
    if ao is not None:
        bpy.context.scene.world.light_settings.use_ambient_occlusion = ao

    # Color mode of rendering
    if color_mode is not None:
        scene.render.image_settings.color_mode = color_mode

    # File format of the render
    if file_format is not None:
        scene.render.image_settings.file_format = file_format

    # Color depth of rendering
    if color_depth is not None:
        scene.render.image_settings.color_depth = color_depth


def _render_prepare(cam, obj_names):
    if cam is None:
        cams = [o for o in bpy.data.objects if o.type == 'CAMERA']
        assert (len(cams) == 1), ("There should be exactly one camera in the scene, "
                                  "when 'cam' is not given")
        cam = cams[0]

    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in bpy.data.objects if o.type == 'MESH']

    scene = bpy.context.scene

    # Set active camera
    scene.camera = cam

    # Hide objects to ignore
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = obj.name not in obj_names

    scene.use_nodes = True

    # Clear the current scene node tree to avoid unexpected renderings
    nodes = scene.node_tree.nodes
    for n in nodes:
        if n.name != "Render Layers":
            nodes.remove(n)

    outnode = nodes.new('CompositorNodeOutputFile')

    return cam.name, obj_names, scene, outnode


def _render(scene, outnode, result_socket, outpath, exr=True, alpha=True):
    node_tree = scene.node_tree

    # Set output file format
    if exr:
        file_format = 'OPEN_EXR'
        color_depth = '32'
        ext = '.exr'
    else:
        file_format = 'PNG'
        color_depth = '16'
        ext = '.png'
    if alpha:
        color_mode = 'RGBA'
    else:
        color_mode = 'RGB'

    outnode.base_path = '/tmp/%s' % time()

    # Connect result socket(s) to the output node
    if isinstance(result_socket, dict):
        assert exr, ".exr must be used for multi-layer results"
        file_format += '_MULTILAYER'

        assert 'composite' in result_socket.keys(), \
            ("Composite pass is always rendered anyways. "
             "Plus, we need this dummy connection for the multi-layer OpenEXR "
             "file to be saved to disk (strangely)")
        node_tree.links.new(result_socket['composite'], outnode.inputs['Image'])

        # Add input slots and connect
        for k, v in result_socket.items():
            outnode.layer_slots.new(k)
            node_tree.links.new(v, outnode.inputs[k])

        render_f = outnode.base_path + '0001.exr'
    else:
        node_tree.links.new(result_socket, outnode.inputs['Image'])

        render_f = join(outnode.base_path, 'Image0001' + ext)

    outnode.format.file_format = file_format
    outnode.format.color_depth = color_depth
    outnode.format.color_mode = color_mode

    scene.render.filepath = '/tmp/%s' % time() # composite (to discard)

    # Render
    bpy.ops.render.render(write_still=True)

    # Move from temporary directory to the desired location
    if not outpath.endswith(ext):
        outpath += ext
    move(render_f, outpath)
    return outpath


def render(outpath, cam=None, obj_names=None, text=None):
    """
    Render current scene to images with cameras in scene

    Args:
        outpath: Path to save render to, e.g., '~/foo.png'
            String
        cam: Camera through which scene is rendered
            bpy_types.Object or None
            Optional; defaults to None (the only camera in scene)
        obj_names: Name(s) of object(s) of interest
            String or list thereof
            Optional; defaults to None (all objects)
        text: What text to be overlaid on image and how
            Dictionary of the following format
            {
                'contents': 'Hello World!',
                'bottom_left_corner': (50, 50),
                'font_scale': 1,
                'bgr': (255, 0, 0),
                'thickness': 2
            }
            Optional; defaults to None
    """
    logger_name = thisfile + '->render()'

    outdir = dirname(outpath)
    if not exists(outdir):
        makedirs(outdir)

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    result_socket = scene.node_tree.nodes['Render Layers'].outputs['Image']

    # Render
    exr = outpath.endswith('.exr')
    outpath = _render(scene, outnode, result_socket, outpath, exr=exr, alpha=False)

    # Optionally overlay text
    if text is not None:
        import cv2
        im = cv2.imread(outpath, cv2.IMREAD_UNCHANGED)
        cv2.putText(im, text['contents'], text['bottom_left_corner'],
                    cv2.FONT_HERSHEY_SIMPLEX, text['font_scale'],
                    text['bgr'], text['thickness'])
        cv2.imwrite(outpath, im)

    logger.name = logger_name
    logger.info("%s rendered through '%s'", obj_names, cam_name)
    logger.warning("    ...; node trees and renderability of these objects have changed")


def render_depth(outprefix, cam=None, obj_names=None, ray_depth=False):
    """
    Render raw (.exr) depth map, in the form of an aliased z map and an anti-aliased alpha map,
        of the specified object(s) from the specified camera

    Args:
        outprefix: Where to save the .exr maps, e.g., '~/depth'
            String
        cam: Camera through which scene is rendered
            bpy_types.Object or None
            Optional; defaults to None (the only camera in scene)
        obj_names: Name(s) of object(s) of interest
            String or list thereof
            Optional; defaults to None (all objects)
        ray_depth: Whether to render ray or plane depth
            Boolean
            Optional; defaults to False (plane depth)
    """
    logger_name = thisfile + '->render_depth()'

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    if ray_depth:
        raise NotImplementedError("Ray depth")

    # Use Blender Render for anti-aliased results -- faster than Cycles,
    # which needs >1 samples to figure out object boundary
    scene.render.engine = 'BLENDER_RENDER'
    scene.render.alpha_mode = 'TRANSPARENT'

    node_tree = scene.node_tree
    nodes = node_tree.nodes

    # Render z pass, without anti-aliasing to avoid values interpolated between
    # real depth values (e.g., 1.5) and large background depth values (e.g., 1e10)
    scene.render.use_antialiasing = False
    scene.use_nodes = True
    try:
        result_socket = nodes['Render Layers'].outputs['Z']
    except KeyError:
        result_socket = nodes['Render Layers'].outputs['Depth']
    outpath_z = _render(scene, outnode, result_socket, outprefix + '_z')

    # Render alpha pass, with anti-aliasing to get a soft mask for blending
    scene.render.use_antialiasing = True
    result_socket = nodes['Render Layers'].outputs['Alpha']
    outpath_a = _render(scene, outnode, result_socket, outprefix + '_a')

    logger.name = logger_name
    logger.info("Depth map of %s rendered through '%s' to", obj_names, cam_name)
    logger.info("    1. z w/o anti-aliasing: %s", outpath_z)
    logger.info("    2. alpha w/ anti-aliasing: %s", outpath_a)
    logger.warning("    ..., and the scene node tree has changed")


def render_mask(outpath, cam=None, obj_names=None, soft=False):
    """
    Render binary or soft mask of objects from the specified camera,
        with bright being the foreground

    Args:
        outpath: Path to save render to, e.g., '~/foo.png'
            String
        cam: Camera through which scene is rendered
            bpy_types.Object or None
            Optional; defaults to None (the only camera in scene)
        obj_names: Name(s) of object(s) of interest
            String or list thereof
            Optional; defaults to None (all objects)
        soft: Whether to render the mask soft or not
            Boolean
            Optional; defaults to False
    """
    logger_name = thisfile + '->render_mask()'

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    if soft:
        scene.render.engine = 'BLENDER_RENDER'
        scene.render.alpha_mode = 'TRANSPARENT'
    else:
        scene.render.engine = 'CYCLES'
        scene.cycles.film_transparent = True
        # Anti-aliased edges are built up by averaging multiple samples
        scene.cycles.samples = 1

    # Set nodes for (binary) alpha pass rendering
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    result_socket = nodes['Render Layers'].outputs['Alpha']

    # Render
    outpath = _render(scene, outnode, result_socket, outpath, exr=False, alpha=False)

    logger.name = logger_name
    logger.info("Mask image of %s rendered through '%s'", obj_names, cam_name)
    logger.warning("    ...; node trees and renderability of these objects have changed")


def render_normal(outpath, cam=None, obj_names=None, camera_space=True):
    """
    Render raw (.exr) normal map of the specified object(s) from the specified camera
        RGB at each pixel is the (almost unit) normal vector at that location

    Args:
        outpath: Where to save the .exr (i.e., raw) normal map
            String
        cam: Camera through which scene is rendered
            bpy_types.Object or None
            Optional; defaults to None (the only camera in scene)
        obj_names: Name(s) of object(s) of interest
            String or list thereof. Use 'ref-ball' for reference normal ball
            Optional; defaults to None (all objects)
        camera_space: Whether to render normal in the camera or world space
            Boolean
            Optional; defaults to True
    """
    from xiuminglib.blender.object import add_sphere
    from xiuminglib.blender.camera import point_camera_to, get_2d_bounding_box

    logger_name = thisfile + '->render_normal()'

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    # # Make normals consistent
    # for obj_name in obj_names:
    #     scene.objects.active = bpy.data.objects[obj_name]
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all()
    #     bpy.ops.mesh.normals_make_consistent()
    #     bpy.ops.object.mode_set(mode='OBJECT')

    # Add reference normal ball
    if 'ref-ball' in obj_names:
        world_origin = (0, 0, 0)
        sphere = add_sphere(location=world_origin)
        point_camera_to(cam, world_origin, up=(0, 0, 1)) # point camera to there
        # Decide scale of the ball so that it, when projected, fits into the frame
        bbox = get_2d_bounding_box(sphere, cam)
        s = max((bbox[1, 0] - bbox[0, 0]) / scene.render.resolution_x,
                (bbox[3, 1] - bbox[0, 1]) / scene.render.resolution_y) * 1.2
        sphere.scale = (1 / s, 1 / s, 1 / s)

    # Set up scene node tree
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    scene.render.layers['RenderLayer'].use_pass_normal = True
    set_alpha_node = nodes.new('CompositorNodeSetAlpha')
    node_tree.links.new(nodes['Render Layers'].outputs['Alpha'],
                        set_alpha_node.inputs['Alpha'])
    node_tree.links.new(nodes['Render Layers'].outputs['Normal'],
                        set_alpha_node.inputs['Image'])
    result_socket = set_alpha_node.outputs['Image']

    # Select rendering engine based on whether camera or object space is desired
    if camera_space:
        scene.render.engine = 'BLENDER_RENDER'
        scene.render.alpha_mode = 'TRANSPARENT'
    else:
        scene.render.engine = 'CYCLES'
        scene.cycles.film_transparent = True
        scene.cycles.samples = 16 # for anti-aliased edges

    # Render
    outpath = _render(scene, outnode, result_socket, outpath)

    logger.name = logger_name
    logger.info("Normal map of %s rendered through '%s' to %s", obj_names, cam_name, outpath)
    logger.warning("    ..., and the scene node tree has changed")


def render_lighting_passes(outpath, cam=None, obj_names=None, n_samples=64):
    """
    Render select Cycles' lighting passes of the specified object(s)
        from the specified camera, into a single multi-layer .exr file

    Args:
        outpath: Where to save the lighting passes
            String
        cam: Camera through which scene is rendered
            bpy_types.Object or None
            Optional; defaults to None (the only camera in scene)
        obj_names: Name(s) of object(s) of interest
            String or list thereof
            Optional; defaults to None (all objects)
        n_samples: Number of path tracing samples per pixel
            Natural number
            Optional; defaults to 64
    """
    logger_name = thisfile + '->render_lighting_passes()'

    select_passes = [
        'diffuse_direct', 'diffuse_indirect', 'diffuse_color',
        'glossy_direct', 'glossy_indirect', 'glossy_color',
    ] # for the purpose of intrinsic images

    cam_name, obj_names, scene, outnode = _render_prepare(cam, obj_names)

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = n_samples
    scene.cycles.film_transparent = True

    # Enable all passes of interest
    render_layer = scene.render.layers['RenderLayer']
    node_tree = scene.node_tree
    nodes = node_tree.nodes
    result_sockets = {
        'composite': nodes['Render Layers'].outputs['Image'],
    }
    for p in select_passes:
        setattr(render_layer, 'use_pass_' + p, True)
        p_key = ' '.join(x.capitalize() for x in p.split('_'))
        # Set alpha
        set_alpha_node = nodes.new('CompositorNodeSetAlpha')
        node_tree.links.new(nodes['Render Layers'].outputs['Alpha'],
                            set_alpha_node.inputs['Alpha'])
        node_tree.links.new(nodes['Render Layers'].outputs[p_key],
                            set_alpha_node.inputs['Image'])
        result_sockets[p] = set_alpha_node.outputs['Image']

    # Render
    outpath = _render(scene, outnode, result_sockets, outpath)

    logger.name = logger_name
    logger.info("Select lighting passes of %s rendered through '%s' to %s", obj_names, cam_name, outpath)
    logger.warning("    ..., and the scene node tree has changed")
