try:
    import bpy
except ModuleNotFoundError:
    pass


def get(otype, any_ok=False):
    """Gets the handle of the only (or any) object of the given type.

    Args:
        otype (str): Object type: ``'MESH'``, ``'CAMERA'``, ``'LAMP'`` or any
            string ``a_bpy_obj.type`` may return.
        any_ok (bool, optional): Whether it's ok to grab any object when there
            exist multiple ones matching the given type. If ``False``, there
            must be exactly one object of the given type.

    Raises:
        RuntimeError: If it's ambiguous which object to get.

    Returns:
        bpy_types.Object.
    """
    objs = [x for x in bpy.data.objects if x.type == otype]
    n_objs = len(objs)
    if n_objs == 0:
        raise RuntimeError("There's no object matching the given type")
    if n_objs == 1:
        return objs[0]
    # More than one objects
    if any_ok:
        return objs[0]
    raise RuntimeError(("When `any_ok` is `False`, there must be exactly "
                        "one object matching the given type"))
