import re
from os.path import abspath, basename, dirname
import numpy as np
try:
    import bpy
    import bmesh
    from mathutils import Matrix, Vector
except ModuleNotFoundError:
    # For building the doc
    pass

from .. import config, os as xm_os
logger, thisfile = config.create_logger(abspath(__file__))


def remove_objects(name_pattern, regex=False):
    """Removes object(s) from current scene.

    Args:
        name_pattern (str): Name or name pattern of object(s) to remove.
        regex (bool, optional): Whether to interpret ``name_pattern`` as a regex.
    """
    logger_name = thisfile + '->remove_objects()'

    objs = bpy.data.objects
    removed = []

    if regex:
        assert (name_pattern != '*'), "Want to match everything? Correct regex for this is '.*'"

        name_pattern = re.compile(name_pattern)

        for obj in objs:
            if name_pattern.match(obj.name):
                obj.select = True
                removed.append(obj.name)
            else:
                obj.select = False

    else:
        for obj in objs:
            if obj.name == name_pattern:
                obj.select = True
                removed.append(obj.name)
            else:
                obj.select = False

    # Delete
    bpy.ops.object.delete()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    logger.name = logger_name
    logger.info("Removed from scene: %s", removed)


def import_object(model_path,
                  axis_forward='-Z', axis_up='Y',
                  rot_mat=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                  trans_vec=(0, 0, 0),
                  scale=1,
                  merge=False,
                  name=None):
    """Imports external object to current scene, the low-level way.

    Args:
        model_path (str): Path to object to add.
        axis_forward (str, optional): Which direction is forward.
        axis_up (str, optional): Which direction is upward.
        rot_mat (array_like, optional): 3-by-3 rotation matrix *preceding* translation.
        trans_vec (array_like, optional): 3D translation vector *following* rotation.
        scale (float, optional): Scale of the object.
        merge (bool, optional): Whether to merge objects into one.
        name (str, optional): Object name after import.

    Raises:
        NotImplementedError: If the model is not a .obj or .ply file.

    Returns:
        bpy_types.Object or list(bpy_types.Object): Imported object(s).
    """
    logger_name = thisfile + '->import_object()'

    # Deselect all
    for o in bpy.data.objects:
        o.select = False

    # Import
    if model_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=model_path, axis_forward=axis_forward, axis_up=axis_up)
    elif model_path.endswith('.ply'):
        bpy.ops.import_mesh.ply(filepath=model_path)

        logger.name = logger_name
        logger.warning("axis_forward and axis_up ignored for .ply")
    else:
        raise NotImplementedError(".%s" % model_path.split('.')[-1])

    # Merge, if asked to
    if merge and len(bpy.context.selected_objects) > 1:
        objs_to_merge = bpy.context.selected_objects
        context = bpy.context.copy()
        context['active_object'] = objs_to_merge[0]
        context['selected_objects'] = objs_to_merge
        context['selected_editable_bases'] = [bpy.context.scene.object_bases[o.name] for o in objs_to_merge]
        bpy.ops.object.join(context)
        objs_to_merge[0].name = 'merged' # change object name
        # objs_to_merge[0].data.name = 'merged' # change mesh name

    obj_list = []
    for i, obj in enumerate(bpy.context.selected_objects):

        # Rename
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # Compute world matrix
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # don't scale here
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4

        # Scale
        obj.scale = (scale, scale, scale)

        obj_list.append(obj)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    logger.name = logger_name
    logger.info("Imported: %s", model_path)

    if len(obj_list) == 1:
        return obj_list[0]
    return obj_list


def export_object(obj_names, model_path, axis_forward='-Z', axis_up='Y'):
    """Exports Blender object(s) to a .obj file.

    Args:
        obj_names (str or list(str)): Object name(s) to export.
        model_path (str): Output path ending with .obj.
        axis_forward (str, optional): Which direction is forward.
        axis_up (str, optional): Which direction is upward.

    Raises:
        NotImplementedError: If the output path doesn't end with .obj.

    Writes
        - Exported .obj file, possibly accompanied by a .mtl file.
    """
    logger_name = thisfile + '->export_object()'

    if not model_path.endswith('.obj'):
        raise NotImplementedError(".%s" % model_path.split('.')[-1])

    out_dir = dirname(model_path)
    xm_os.makedirs(out_dir)

    if isinstance(obj_names, str):
        obj_names = [obj_names]

    exported = []
    for o in [x for x in bpy.data.objects if x.type == 'MESH']:
        o.select = o.name in obj_names
        if o.select:
            exported.append(o.name)

    bpy.ops.export_scene.obj(
        filepath=model_path, use_selection=True,
        axis_forward=axis_forward, axis_up=axis_up)

    logger.name = logger_name
    logger.info("%s Exported to %s", exported, model_path)


def add_cylinder_between(pt1, pt2, r, name=None):
    """Adds a cylinder specified by two end points and radius.

    Super useful for visualizing rays in ray tracing while debugging.

    Args:
        pt1 (array_like): Global coordinates of point 1.
        pt2 (array_like): Global coordinates of point 2.
        r (float): Cylinder radius.
        name (str, optional): Cylinder name.

    Returns:
        bpy_types.Object: Cylinder added.
    """
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    d = pt2 - pt1

    # Add cylinder at the correct location
    dist = np.linalg.norm(d)
    loc = (pt1[0] + d[0] / 2, pt1[1] + d[1] / 2, pt1[2] + d[2] / 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=dist, location=loc)

    cylinder_obj = bpy.context.object

    if name is not None:
        cylinder_obj.name = name

    # Further rotate it accordingly
    phi = np.arctan2(d[1], d[0])
    theta = np.arccos(d[2] / dist)
    cylinder_obj.rotation_euler[1] = theta
    cylinder_obj.rotation_euler[2] = phi

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return cylinder_obj


def add_rectangular_plane(center_loc=(0, 0, 0), point_to=(0, 0, 1), size=(2, 2), name=None):
    """Adds a rectangular plane specified by its center location, dimensions, and where its +z points to.

    Args:
        center_loc (array_like, optional): Plane center location in world coordinates.
        point_to (array_like, optional): Point in world coordinates to which plane's +z points.
        size (array_like, optional): Sizes in x and y directions (0 in z).
        name (str, optional): Plane name.

    Returns:
        bpy_types.Object: Plane added.
    """
    center_loc = np.array(center_loc)
    point_to = np.array(point_to)
    size = np.append(np.array(size), 0)

    bpy.ops.mesh.primitive_plane_add(location=center_loc)

    plane_obj = bpy.context.object

    if name is not None:
        plane_obj.name = name

    plane_obj.dimensions = size

    # Point it to target
    direction = Vector(point_to) - plane_obj.location
    # Find quaternion that rotates plane's 'Z' so that it aligns with `direction`
    # This rotation is not unique because the rotated plane can still rotate about direction vector
    # Specifying 'Y' gives the rotation quaternion with plane's 'Y' pointing up
    rot_quat = direction.to_track_quat('Z', 'Y')
    plane_obj.rotation_euler = rot_quat.to_euler()

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return plane_obj


def create_mesh(verts, faces, name):
    """Creates a mesh from vertices and faces.

    Args:
        verts (array_like): Local coordinates of the vertices. Of shape N-by-3.
        faces (list(tuple)): Faces specified by ordered vertex indices.
        name (str): Mesh name.

    Returns:
        bpy_types.Mesh: Mesh data created.
    """
    logger_name = thisfile + '->create_mesh()'

    verts = np.array(verts)

    # Create mesh
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    logger.name = logger_name
    logger.info("Mesh '%s' created", name)

    return mesh_data


def create_object_from_mesh(mesh_data, obj_name, location=(0, 0, 0), rotation_euler=(0, 0, 0), scale=(1, 1, 1)):
    """Creates object from mesh data.

    Args:
        mesh_data (bpy_types.Mesh): Mesh data.
        obj_name (str): Object name.
        location (tuple, optional): Object location in world coordinates.
        rotation_euler (tuple, optional): Object rotation in radians.
        scale (tuple, optional): Object scale.

    Returns:
        bpy_types.Object: Object created.
    """
    logger_name = thisfile + '->create_object_from_mesh()'

    # Create
    obj = bpy.data.objects.new(obj_name, mesh_data)

    # Link to current scene
    scene = bpy.context.scene
    scene.objects.link(obj)
    obj.select = True
    scene.objects.active = obj # make the selection effective

    # Set attributes
    obj.location = location
    obj.rotation_euler = rotation_euler
    obj.scale = scale

    logger.name = logger_name
    logger.info("Object '%s' created from mesh data and selected", obj_name)

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    return obj


def _clear_nodetree_for_active_material(obj):
    """Internal helper function clears the node tree of active material.

    So that desired node tree can be cleanly set up. If no active material, one will be created.
    """
    # Create material if none
    if obj.active_material is None:
        mat = bpy.data.materials.new(name='new-mat-for-%s' % obj.name)
        if obj.data.materials:
            # Assign to first material slot
            obj.data.materials[0] = mat
        else:
            # No slots
            obj.data.materials.append(mat)

    active_mat = obj.active_material
    active_mat.use_nodes = True
    node_tree = active_mat.node_tree
    nodes = node_tree.nodes

    # Remove all nodes
    for node in nodes:
        nodes.remove(node)

    return node_tree


def color_vertices(obj, vert_ind, colors):
    r"""Colors each vertex of interest with the given color.

    Colors are defined for vertex loops, in fact. This function uses the same color
    for all loops of a vertex. Useful for making a 3D heatmap.

    Args:
        obj (bpy_types.Object): Object.
        vert_ind (int or list(int)): Index/indices of vertex/vertices to color.
        colors (tuple or list(tuple)): RGB value(s) to paint on vertex/vertices.
            Values :math:`\in [0, 1]`. If one tuple, this color will be applied to all vertices.
            If list of tuples, must be of the same length as ``vert_ind``.

    Raises:
        ValueError: If color length is wrong.
    """
    logger_name = thisfile + '->color_vertices()'

    # Validate inputs
    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]
    else:
        vert_ind = list(vert_ind)
    if isinstance(colors, tuple):
        colors = [colors] * len(vert_ind)
    assert (len(colors) == len(vert_ind)), \
        "`colors` and `vert_ind` must be of the same length, or `colors` is a single tuple"
    for i, c in enumerate(colors):
        c = tuple(c)
        if len(c) == 3:
            colors[i] = c + (1,)
        elif len(c) == 4: # In case some Blender version needs 4-tuples
            colors[i] = c
        else:
            raise ValueError("Wrong color length: %d" % len(c))
    if any(x > 1 for c in colors for x in c):
        logger.warning("Did you forget to normalize color values to [0, 1]?")

    scene = bpy.context.scene
    scene.objects.active = obj
    obj.select = True
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data

    if mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.active
    else:
        vcol_layer = mesh.vertex_colors.new()

    # A vertex and one of its edges combined are called a loop, which has a color
    # So if a vertex has four outgoing edges, it has four colors for the four loops
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop_vert_idx = mesh.loops[loop_idx].vertex_index
            try:
                color_idx = vert_ind.index(loop_vert_idx)
            except ValueError:
                color_idx = None
            if color_idx is not None:
                try:
                    vcol_layer.data[loop_idx].color = colors[color_idx]
                except ValueError:
                    # This Blender version requires 3-tuples
                    vcol_layer.data[loop_idx].color = colors[color_idx][:3]

    # Set up nodes for vertex colors
    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes
    nodes.new('ShaderNodeAttribute')
    nodes.new('ShaderNodeBsdfDiffuse')
    nodes.new('ShaderNodeOutputMaterial')
    nodes['Attribute'].attribute_name = vcol_layer.name
    node_tree.links.new(nodes['Attribute'].outputs[0], nodes['Diffuse BSDF'].inputs[0])
    node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Vertex color(s) added to '%s'", obj.name)
    logger.warning("    ..., so node tree of '%s' has changed", obj.name)


def _assert_cycles(scene):
    """
    Raises:
        NotImplementedError: If rendering engine is not Cycles.
    """
    engine = scene.render.engine
    if engine != 'CYCLES':
        raise NotImplementedError(engine)


def _make_texture_node(obj, texture_str):
    mat = obj.active_material
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    nodes.new('ShaderNodeTexImage')
    texture_node = nodes['Image Texture']
    if texture_str == 'bundled':
        texture = mat.active_texture
        assert texture is not None, "No bundled texture found"
        img = texture.image
    else:
        # Path given -- external texture map
        bpy.data.images.load(texture_str, check_existing=True)
        img = bpy.data.images[basename(texture_str)]
        nodes.new('ShaderNodeTexCoord') # careless
        node_tree.links.new(nodes['Texture Coordinate'].outputs['Generated'],
                            texture_node.inputs['Vector'])
    texture_node.image = img
    return texture_node


def setup_diffuse_nodetree(obj, texture, roughness=0):
    r"""Sets up a diffuse texture node tree.

    Bundled texture can be an external texture map (carelessly mapped) or a pure color.
    Mathematically, the BRDF model used is either Lambertian (no roughness) or Oren-Nayar (with roughness).

    Args:
        obj (bpy_types.Object): Object, optionally bundled with texture map.
        texture (str or tuple): If string, must be ``'bundled'`` or path to the texture image.
            If tuple, must be of 4 floats :math:`\in [0, 1]` as RGBA values.
        roughness (float, optional): Roughness in Oren-Nayar model. 0 gives Lambertian.

    Raises:
        TypeError: If ``texture`` is of wrong type.
    """
    logger_name = thisfile + '->setup_diffuse_nodetree()'

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    # Set color for diffuse node
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    if isinstance(texture, str):
        texture_node = _make_texture_node(obj, texture)
        node_tree.links.new(texture_node.outputs['Color'], diffuse_node.inputs['Color'])
    elif isinstance(texture, tuple):
        diffuse_node.inputs['Color'].default_value = texture
    else:
        raise TypeError(texture)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Roughness
    diffuse_node.inputs['Roughness'].default_value = roughness

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Diffuse node tree set up for '%s'", obj.name)


def setup_glossy_nodetree(obj, color=(1, 1, 1, 1), roughness=0):
    r"""Sets up a glossy node tree for a pure color.

    To extend it with texture maps, see :func:`setup_diffuse_nodetree`.

    Args:
        obj (bpy_types.Object): Object bundled with texture map.
        color (tuple, optional): RGBA values :math:`\in [0, 1]`.
        roughness (float, optional): Roughness. 0 means perfectly reflective.
    """
    logger_name = thisfile + '->setup_glossy_nodetree()'

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    nodes.new('ShaderNodeBsdfGlossy')
    nodes['Glossy BSDF'].inputs[0].default_value = color
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(nodes['Glossy BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # Roughness
    node_tree.nodes['Glossy BSDF'].inputs['Roughness'].default_value = roughness

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Glossy node tree set up for '%s'", obj.name)


def setup_emission_nodetree(obj, color=(1, 1, 1, 1), strength=1):
    r"""Sets up an emission node tree for the object.

    Args:
        obj (bpy_types.Object): Object bundled with texture map.
        color (tuple, optional): Emission RGBA :math:`\in [0, 1]`.
        strength (float, optional): Emission strength.
    """
    logger_name = thisfile + '->setup_emission_nodetree()'

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    nodes.new('ShaderNodeEmission')
    nodes['Emission'].inputs[0].default_value = color
    nodes['Emission'].inputs[1].default_value = strength
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(nodes['Emission'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Emission node tree set up for '%s'", obj.name)


def setup_holdout_nodetree(obj):
    """Sets up a holdout node tree for the object.

    Args:
        obj (bpy_types.Object): Object bundled with texture map.
    """
    logger_name = thisfile + '->setup_holdout_nodetree()'

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    nodes.new('ShaderNodeHoldout')
    nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(nodes['Holdout'].outputs[0], nodes['Material Output'].inputs[0])

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Holdout node tree set up for '%s'", obj.name)


def setup_retroreflective_nodetree(obj, texture, roughness=0, glossy_weight=0.1):
    r"""Sets up a retroreflective texture node tree.

    Bundled texture can be an external texture map (carelessly mapped) or a pure color.
    Mathematically, the BRDF model is a mixture of a diffuse BRDF and a glossy BRDF using
    incoming light directions as normals.

    Args:
        obj (bpy_types.Object): Object, optionally bundled with texture map.
        texture (str or tuple): If string, must be ``'bundled'`` or path to the texture image.
            If tuple, must be of 4 floats :math:`\in [0, 1]` as RGBA values.
        roughness (float, optional): Roughness for both the glossy and diffuse shaders.
        glossy_weight (float, optional): Mixture weight for the glossy shader.

    Raises:
        TypeError: If ``texture`` is of wrong type.
    """
    logger_name = thisfile + '->setup_retroreflective_nodetree()'

    scene = bpy.context.scene
    _assert_cycles(scene)

    node_tree = _clear_nodetree_for_active_material(obj)
    nodes = node_tree.nodes

    # Set color for diffuse and glossy nodes
    diffuse_node = nodes.new('ShaderNodeBsdfDiffuse')
    glossy_node = nodes.new('ShaderNodeBsdfGlossy')
    if isinstance(texture, str):
        texture_node = _make_texture_node(obj, texture)
        node_tree.links.new(texture_node.outputs['Color'], diffuse_node.inputs['Color'])
        node_tree.links.new(texture_node.outputs['Color'], glossy_node.inputs['Color'])
    elif isinstance(texture, tuple):
        diffuse_node.inputs['Color'].default_value = texture
        glossy_node.inputs['Color'].default_value = texture
    else:
        raise TypeError(texture)

    geometry_node = nodes.new('ShaderNodeNewGeometry')
    mix_node = nodes.new('ShaderNodeMixShader')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    node_tree.links.new(geometry_node.outputs['Incoming'], glossy_node.inputs['Normal'])
    node_tree.links.new(diffuse_node.outputs['BSDF'], mix_node.inputs[1])
    node_tree.links.new(glossy_node.outputs['BSDF'], mix_node.inputs[2])
    node_tree.links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])

    # Roughness
    diffuse_node.inputs['Roughness'].default_value = roughness
    glossy_node.inputs['Roughness'].default_value = roughness

    mix_node.inputs['Fac'].default_value = glossy_weight

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Retroreflective node tree set up for '%s'", obj.name)


def get_bmesh(obj):
    """Gets Blender mesh data from object.

    Args:
        obj (bpy_types.Object): Object.

    Returns:
        BMesh: Blender mesh data.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Scene update necessary, as matrix_world is updated lazily
    bpy.context.scene.update()

    return bm


def subdivide_mesh(obj, n_subdiv=2):
    """Subdivides mesh of object.

    Args:
        obj (bpy_types.Object): Object whose mesh is to be subdivided.
        n_subdiv (int, optional): Number of subdivision levels.
    """
    logger_name = thisfile + '->subdivide_mesh()'

    scene = bpy.context.scene

    # All objects need to be in 'OBJECT' mode to apply modifiers -- maybe a Blender bug?
    for o in bpy.data.objects:
        scene.objects.active = o
        bpy.ops.object.mode_set(mode='OBJECT')
        o.select = False
    obj.select = True
    scene.objects.active = obj

    bpy.ops.object.modifier_add(type='SUBSURF')
    obj.modifiers['Subsurf'].subdivision_type = 'CATMULL_CLARK'
    obj.modifiers['Subsurf'].levels = n_subdiv
    obj.modifiers['Subsurf'].render_levels = n_subdiv

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier='Subsurf', apply_as='DATA')

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Subdivided mesh of '%s'", obj.name)


def select_mesh_elements_by_vertices(obj, vert_ind, select_type):
    """Selects vertices or their associated edges/faces in edit mode.

    Args:
        obj (bpy_types.Object): Object.
        vert_ind (int or list(int)): Vertex index/indices.
        select_type (str): Type of mesh elements to select: ``'vertex'``, ``'edge'`` or ``'face'``.

    Raises:
        ValueError: If ``select_type`` value is invalid.
    """
    logger_name = thisfile + '->select_mesh_elements_by_vertices()'

    if isinstance(vert_ind, int):
        vert_ind = [vert_ind]

    # Edit mode
    scene = bpy.context.scene
    scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Deselect all
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')

    bm = bmesh.from_edit_mesh(obj.data)
    bvs = bm.verts

    bvs.ensure_lookup_table()
    for i in vert_ind:
        bv = bvs[i]

        if select_type == 'vertex':
            bv.select = True

        # Select all edges with this vertex at an end
        elif select_type == 'edge':
            for be in bv.link_edges:
                be.select = True

        # Select all faces with this vertex
        elif select_type == 'face':
            for bf in bv.link_faces:
                bf.select = True

        else:
            raise ValueError("Wrong selection type")

    # Update viewport
    scene.objects.active = scene.objects.active

    # Scene update necessary, as matrix_world is updated lazily
    scene.update()

    logger.name = logger_name
    logger.info("Selected %s elements of '%s'", select_type, obj.name)


def add_sphere(location=(0, 0, 0), scale=1, n_subdiv=4):
    """Adds a sphere.

    Args:
        location (array_like, optional): Location of the sphere center.
        scale (float, optional): Scale of the sphere.
        n_subdiv (int, optional): Control of how round the sphere is.

    Returns:
        bpy_types.Object: Sphere created.
    """
    bpy.ops.mesh.primitive_ico_sphere_add()
    sphere = bpy.context.active_object

    sphere.location = location
    sphere.scale = (scale, scale, scale)

    # Subdivide for smoother sphere
    bpy.ops.object.modifier_add(type='SUBSURF')
    sphere.modifiers['Subsurf'].subdivision_type = 'CATMULL_CLARK'
    sphere.modifiers['Subsurf'].levels = n_subdiv
    sphere.modifiers['Subsurf'].render_levels = n_subdiv
    bpy.context.scene.objects.active = sphere
    bpy.ops.object.modifier_apply(modifier='Subsurf', apply_as='DATA')

    return sphere
