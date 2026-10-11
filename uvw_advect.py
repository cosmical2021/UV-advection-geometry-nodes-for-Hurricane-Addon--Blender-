"""
UVW Advect - Geometry Nodes + Shader node group for advected (rest-position) UVW on dynamic meshes.

Run this file from Blender's Text Editor (Run Script) or install it as an add-on.
It creates two node groups in the current .blend (rebuilding them in place if they already exist):

  "UVW Advect"         Geometry Nodes group (modifier). Transfers a per-particle rest position
                       (Hurricane: "hurr_rest_position") from a point cloud onto the vertices of a
                       mesh with changing topology and stores it as the point attribute "uvw".
                       Optionally also writes a triplanar UV map ("uvw_triplanar", face corner
                       domain): every face is projected along the dominant axis of its rest-space
                       normal, so it works with plain 2D image textures and any UV workflow.
  "UVW Advect Shader"  Shader node group. Reads the "uvw" attribute and outputs a UVW vector that
                       you can plug into any shader node, with optional random UVW islands, and
                       the "Triplanar UV" output for the triplanar UV map.
  "UVW Blur"           Shader node group. Blur feature: the Geometry Nodes group also computes a
                       smoothed copy of the uvw map (Color Blur Iterations input) and stores the
                       difference sharp-minus-blurred as "uvw_blur_delta". This group has a single
                       UVW slot - connect any UVW source (UVW Advect Shader, Triplanar UV, a
                       Mapping node, ...) - and mixes it subtractively with the blurred map:
                       out = UVW - Blur * delta. The mapping is pulled toward the smoothed field,
                       so texture colors smear/mix together.

  "UVW States"         Shader node group. Projected states + morph: pick up to 3 "State" mesh objects
                       on the modifier. Each is a copy of the liquid at the rest frame (same world
                       position) whose UV map you adjusted by hand. Every liquid vertex looks up the
                       nearest point on each state mesh, in rest space, and takes that state's UV
                       ("uvw_state1..3"), so the projected UVs follow the liquid as it moves. The
                       "Morph" value (0 = state 1, 1 = state 2, 2 = state 3) blends between them:
                       "UVW" is the blended UV (the texture slides from one state's mapping to the
                       next), "UVW 1..3" + "Weight 1..3" let you use one texture per state and
                       cross-fade them ("test 1" fades into "test 2"). Use Flat projection textures.

Cube projection: enable "Cube Projection" and pick a "Projection Object" (or run F3 -> "UVW Advect:
Add Projection Cube" to create an Empty fitted to the surface). The start UVW becomes the rest
position in that object's local space, with the cube spanning 0-1. Place it on the first frame
and keep it still; the UVW then stays attached to the fluid.

Quick start: select the surface mesh, then shift-select the point cloud (active = surface),
F3 -> "UVW Advect: Setup".  Or do it by hand: add a Geometry Nodes modifier with "UVW Advect",
pick the point cloud object, and add the "UVW Advect Shader" group node to your material.
"""

bl_info = {
    "name": "UVW Advect",
    "author": "UVW",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "F3 search: UVW Advect",
    "description": "Advected UVW from particle rest positions for dynamic-topology meshes",
    "category": "Node",
}

import math
import os

import bpy

GN_NAME = "UVW Advect"
SHADER_NAME = "UVW Advect Shader"
BLUR_NAME = "UVW Blur"
STATES_NAME = "UVW States"
NUM_STATES = 3
STATE_ATTR = "uvw_state%d"
UVW_ATTR = "uvw"
BLUR_ATTR = "uvw_blur_delta"
TRIPLANAR_ATTR = "uvw_triplanar"
HURRICANE_REST_ATTR = "hurr_rest_position"


# --------------------------------------------------------------------------- helpers
def _node_trees():
    yield from bpy.data.node_groups
    for coll in (bpy.data.materials, bpy.data.worlds, bpy.data.lights):
        for idb in coll:
            if idb.node_tree is not None:
                yield idb.node_tree


def _snapshot_instances(name):
    """Input values and links of every group node using the group `name`.

    Rebuilding a group in place resets the values on existing group nodes to 0 and drops the
    links on its sockets, so they are captured here and put back by _restore_instances."""
    snap = []
    for nt in _node_trees():
        for n in nt.nodes:
            if n.bl_idname == "ShaderNodeGroup" and n.node_tree is not None and n.node_tree.name == name:
                vals = {}
                for s in n.inputs:
                    if hasattr(s, "default_value") and not s.is_linked:
                        v = s.default_value
                        vals[s.name] = tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
                links = []
                for l in nt.links:
                    if l.to_node == n:
                        links.append(("in", l.to_socket.name, l.from_node.name, l.from_socket.name))
                    elif l.from_node == n:
                        links.append(("out", l.from_socket.name, l.to_node.name, l.to_socket.name))
                snap.append((nt, n.name, vals, links))
    return snap


def _restore_instances(snap):
    for nt, node_name, vals, links in snap:
        n = nt.nodes[node_name]
        for s in n.inputs:
            if s.name in vals and hasattr(s, "default_value"):
                try:
                    s.default_value = vals[s.name]
                except (TypeError, ValueError, AttributeError):
                    pass
        for direction, own, other_name, other in links:
            other_node = nt.nodes.get(other_name)
            if other_node is None:
                continue
            try:
                if direction == "in":
                    nt.links.new(other_node.outputs[other], n.inputs[own])
                else:
                    nt.links.new(n.outputs[own], other_node.inputs[other])
            except (KeyError, RuntimeError):
                pass


def _group(name, tree_type):
    ng = bpy.data.node_groups.get(name)
    if ng is None or ng.bl_idname != tree_type:
        ng = bpy.data.node_groups.new(name, tree_type)
    else:
        ng.nodes.clear()
        ng.interface.clear()
    ng.use_fake_user = True
    return ng


def _socket(ng, name, in_out, stype, default=None, **props):
    s = ng.interface.new_socket(name, in_out=in_out, socket_type=stype)
    if default is not None:
        s.default_value = default
    for key, val in props.items():
        setattr(s, key, val)
    return s


def _node(ng, idname, x, y, label=None, parent=None, **props):
    n = ng.nodes.new(idname)
    n.location = (x, y)
    for key, val in props.items():
        setattr(n, key, val)
    if label:
        n.label = label
    if parent is not None:
        n.parent = parent
    return n


def _frame(ng, label, color=None):
    f = ng.nodes.new("NodeFrame")
    f.label = label
    f.label_size = 18
    if color:
        f.use_custom_color = True
        f.color = color
    return f


def _sock(sockets, *names):
    """First enabled socket matching one of the names (typed socket variants share a name)."""
    for name in names:
        for s in sockets:
            if s.name == name and s.enabled:
                return s
    raise KeyError(names)


def _link(ng, a, b):
    ng.links.new(a, b)


def _vmath(ng, op, x, y, a=None, b=None, scale=None, parent=None, label=None):
    n = _node(ng, "ShaderNodeVectorMath", x, y, label=label, parent=parent, operation=op)
    if a is not None:
        if isinstance(a, bpy.types.NodeSocket):
            _link(ng, a, n.inputs[0])
        else:
            n.inputs[0].default_value = a
    if b is not None:
        if isinstance(b, bpy.types.NodeSocket):
            _link(ng, b, n.inputs[1])
        else:
            n.inputs[1].default_value = b
    if scale is not None:
        if isinstance(scale, bpy.types.NodeSocket):
            _link(ng, scale, n.inputs[3])
        else:
            n.inputs[3].default_value = scale
    return n


def _math(ng, op, x, y, a=None, b=None, parent=None, label=None):
    n = _node(ng, "ShaderNodeMath", x, y, label=label, parent=parent, operation=op)
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, bpy.types.NodeSocket):
            _link(ng, v, n.inputs[i])
        else:
            n.inputs[i].default_value = v
    return n


# --------------------------------------------------------------------------- geometry nodes
def build_geometry_nodes():
    ng = _group(GN_NAME, "GeometryNodeTree")
    ng.is_modifier = True

    _socket(ng, "Geometry", "OUTPUT", "NodeSocketGeometry")
    _socket(ng, "Geometry", "INPUT", "NodeSocketGeometry",
            description="Surface mesh (any topology, may change every frame)")
    _socket(ng, "Particles", "INPUT", "NodeSocketObject",
            description="Point cloud / particle object that carries the rest position attribute")
    _socket(ng, "Rest Attribute", "INPUT", "NodeSocketString", HURRICANE_REST_ATTR,
            description="Per-particle rest position attribute. If missing, particle position is used")
    _socket(ng, "UVW Attribute", "INPUT", "NodeSocketString", UVW_ATTR,
            description="Name of the point attribute written on the mesh "
                        "(the UVW Advect Shader group reads '%s')" % UVW_ATTR)
    _socket(ng, "Cube Projection", "INPUT", "NodeSocketBool", False,
            description="Define the start UVW with the Projection Object: UVW is the rest position in that "
                        "object's local space, so the cube spans 0-1 (one texture tile per cube edge). "
                        "Position the object on the first frame and keep it still")
    _socket(ng, "Projection Object", "INPUT", "NodeSocketObject",
            description="Object whose transform places the cube (an Empty with Cube display, size 1). "
                        "Needs Cube Projection on")
    _socket(ng, "Smooth Taps", "INPUT", "NodeSocketBool", True,
            description="Blend the rest position of several neighbouring particles "
                        "instead of using only the nearest one")
    _socket(ng, "Neighbor Radius", "INPUT", "NodeSocketFloat", 2.0, min_value=0.0, max_value=20.0,
            description="How far around each vertex neighbouring particles are searched, "
                        "relative to the distance to the closest particle")
    _socket(ng, "Blur Iterations", "INPUT", "NodeSocketInt", 2, min_value=0, max_value=100,
            description="Extra smoothing of the result along the mesh edges")
    _socket(ng, "Color Blur Iterations", "INPUT", "NodeSocketInt", 10, min_value=0, max_value=200,
            description="Smoothing of a second UVW map; the difference sharp-minus-blurred is "
                        "stored as '%s'. The '%s' shader group subtracts it from any connected "
                        "UVW map to smear/mix the texture colors" % (BLUR_ATTR, BLUR_NAME))
    for i in range(1, NUM_STATES + 1):
        _socket(ng, "State %d" % i, "INPUT", "NodeSocketObject",
                description="State %d mesh: a copy of the liquid at the rest frame (same world position) "
                            "with a hand-made UV map. Its UVs are projected onto the liquid as '%s'"
                            % (i, STATE_ATTR % i))
    _socket(ng, "State UV Map", "INPUT", "NodeSocketString", "UVMap",
            description="Name of the UV map to read on the state meshes")
    _socket(ng, "Triplanar", "INPUT", "NodeSocketBool", True,
            description="Also write a triplanar UV map: each face is projected along the dominant axis "
                        "of its rest-space normal, so the texture sticks to the surface as it moves")
    _socket(ng, "Triplanar Scale", "INPUT", "NodeSocketFloat", 1.0,
            description="Scale of the triplanar UV map (multiplies the rest-space coordinates)")
    _socket(ng, "Triplanar UV Attribute", "INPUT", "NodeSocketString", TRIPLANAR_ATTR,
            description="Name of the face corner UV map written on the mesh "
                        "(the UVW Advect Shader group reads '%s')" % TRIPLANAR_ATTR)

    gi = _node(ng, "NodeGroupInput", -1400, 0)
    go = _node(ng, "NodeGroupOutput", 2500, 0)

    fr_src = _frame(ng, "Particles + rest position", (0.25, 0.3, 0.4))
    fr_tap = _frame(ng, "Neighbour taps (nearest particle at 7 positions around the vertex, distance weighted)",
                    (0.3, 0.4, 0.3))
    fr_out = _frame(ng, "Smooth + store", (0.4, 0.3, 0.25))

    objinfo = _node(ng, "GeometryNodeObjectInfo", -1100, 200, parent=fr_src, transform_space="RELATIVE")
    _link(ng, gi.outputs["Particles"], objinfo.inputs["Object"])
    particles = objinfo.outputs["Geometry"]

    pos = _node(ng, "GeometryNodeInputPosition", -1100, 0, parent=fr_src)

    named = _node(ng, "GeometryNodeInputNamedAttribute", -1100, -150, parent=fr_src, data_type="FLOAT_VECTOR")
    _link(ng, gi.outputs["Rest Attribute"], named.inputs["Name"])

    rest_sw = _node(ng, "GeometryNodeSwitch", -850, -100, parent=fr_src, label="Rest (or position)",
                    input_type="VECTOR")
    _link(ng, _sock(named.outputs, "Exists", "Found"), _sock(rest_sw.inputs, "Switch"))
    _link(ng, pos.outputs["Position"], _sock(rest_sw.inputs, "False"))
    _link(ng, _sock(named.outputs, "Attribute"), _sock(rest_sw.inputs, "True"))
    rest_raw = _sock(rest_sw.outputs, "Output")

    # ---- cube projection: rest position -> local space of the projection object, remapped so the cube
    # (an Empty with Cube display, size 1) spans 0..1. Place the object on the first frame.
    fr_cube = _frame(ng, "Cube projection (start UVW placed by the Projection Object)", (0.4, 0.35, 0.2))
    cube_info = _node(ng, "GeometryNodeObjectInfo", -1500, -450, parent=fr_cube, transform_space="ORIGINAL",
                      label="Projection Object")
    _link(ng, gi.outputs["Projection Object"], cube_info.inputs["Object"])
    cube_inv = _node(ng, "FunctionNodeInvertMatrix", -1250, -450, parent=fr_cube)
    _link(ng, cube_info.outputs["Transform"], cube_inv.inputs["Matrix"])

    part_info = _node(ng, "GeometryNodeObjectInfo", -1500, -650, parent=fr_cube, transform_space="ORIGINAL",
                      label="Particles (world matrix)")
    _link(ng, gi.outputs["Particles"], part_info.inputs["Object"])
    self_obj = _node(ng, "GeometryNodeSelfObject", -1500, -850, parent=fr_cube)
    self_info = _node(ng, "GeometryNodeObjectInfo", -1250, -850, parent=fr_cube, transform_space="ORIGINAL",
                      label="Surface (world matrix)")
    _link(ng, self_obj.outputs["Self Object"], self_info.inputs["Object"])

    src_mat = _node(ng, "GeometryNodeSwitch", -1250, -650, parent=fr_cube, input_type="MATRIX",
                    label="Space of the rest value")
    _link(ng, _sock(named.outputs, "Exists", "Found"), _sock(src_mat.inputs, "Switch"))
    _link(ng, self_info.outputs["Transform"], _sock(src_mat.inputs, "False"))
    _link(ng, part_info.outputs["Transform"], _sock(src_mat.inputs, "True"))

    to_cube = _node(ng, "FunctionNodeMatrixMultiply", -1000, -550, parent=fr_cube, label="World -> cube space")
    _link(ng, cube_inv.outputs["Matrix"], to_cube.inputs[0])
    _link(ng, _sock(src_mat.outputs, "Output"), to_cube.inputs[1])
    cube_pt = _node(ng, "FunctionNodeTransformPoint", -800, -450, parent=fr_cube)
    _link(ng, rest_raw, cube_pt.inputs["Vector"])
    _link(ng, to_cube.outputs[0], cube_pt.inputs["Transform"])
    cube_01 = _vmath(ng, "MULTIPLY_ADD", -600, -450, a=cube_pt.outputs["Vector"], b=(0.5, 0.5, 0.5),
                     parent=fr_cube, label="Cube spans 0..1")
    cube_01.inputs[2].default_value = (0.5, 0.5, 0.5)

    cube_sw = _node(ng, "GeometryNodeSwitch", -400, -300, parent=fr_cube, input_type="VECTOR",
                    label="Cube Projection?")
    _link(ng, gi.outputs["Cube Projection"], _sock(cube_sw.inputs, "Switch"))
    _link(ng, rest_raw, _sock(cube_sw.inputs, "False"))
    _link(ng, cube_01.outputs["Vector"], _sock(cube_sw.inputs, "True"))
    rest_field = _sock(cube_sw.outputs, "Output")

    def tap(j, query_pos, x, y):
        """Nearest particle to query_pos -> (rest, |P - particle|)."""
        near = _node(ng, "GeometryNodeSampleNearest", x, y, parent=fr_tap, domain="POINT")
        _link(ng, particles, near.inputs["Geometry"])
        _link(ng, query_pos, near.inputs["Sample Position"])
        s_pos = _node(ng, "GeometryNodeSampleIndex", x + 200, y, parent=fr_tap,
                      data_type="FLOAT_VECTOR", domain="POINT", label="Particle pos %d" % j)
        _link(ng, particles, s_pos.inputs["Geometry"])
        _link(ng, pos.outputs["Position"], _sock(s_pos.inputs, "Value"))
        _link(ng, near.outputs["Index"], s_pos.inputs["Index"])
        s_rest = _node(ng, "GeometryNodeSampleIndex", x + 200, y - 180, parent=fr_tap,
                       data_type="FLOAT_VECTOR", domain="POINT", label="Particle rest %d" % j)
        _link(ng, particles, s_rest.inputs["Geometry"])
        _link(ng, rest_field, _sock(s_rest.inputs, "Value"))
        _link(ng, near.outputs["Index"], s_rest.inputs["Index"])
        dist = _vmath(ng, "DISTANCE", x + 400, y, a=pos.outputs["Position"],
                      b=_sock(s_pos.outputs, "Value"), parent=fr_tap)
        return _sock(s_rest.outputs, "Value"), dist.outputs["Value"]

    P = pos.outputs["Position"]
    rest0, d0 = tap(0, P, -600, 600)

    # Shared values derived from the nearest particle distance (keeps everything scale free).
    radius = _math(ng, "MULTIPLY", -600, 150, a=d0, b=gi.outputs["Neighbor Radius"], parent=fr_tap,
                   label="Tap radius")
    soft = _math(ng, "MULTIPLY", -600, 50, a=d0, b=0.25, parent=fr_tap)
    soft2 = _math(ng, "MULTIPLY", -400, 50, a=soft.outputs[0], b=soft.outputs[0], parent=fr_tap)
    eps2 = _math(ng, "ADD", -200, 50, a=soft2.outputs[0], b=1e-20, parent=fr_tap, label="Weight softening")

    def weight(dj, x, y):
        sq = _math(ng, "MULTIPLY", x, y, a=dj, b=dj, parent=fr_tap)
        den = _math(ng, "ADD", x + 150, y, a=sq.outputs[0], b=eps2.outputs[0], parent=fr_tap)
        w = _math(ng, "DIVIDE", x + 300, y, a=1.0, b=den.outputs[0], parent=fr_tap, label="Weight")
        return w.outputs[0]

    w0 = weight(d0, 250, 600)
    wr0 = _vmath(ng, "SCALE", 800, 600, a=rest0, scale=w0, parent=fr_tap)
    acc_v, acc_w = wr0.outputs["Vector"], w0

    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for j, d in enumerate(dirs, start=1):
        x, y = -600, -200 - (j - 1) * 520
        off = _vmath(ng, "SCALE", x, y, a=d, scale=radius.outputs[0], parent=fr_tap)
        q = _vmath(ng, "ADD", x + 200, y, a=P, b=off.outputs["Vector"], parent=fr_tap)
        rj, dj = tap(j, q.outputs["Vector"], x + 400, y)
        wj = weight(dj, x + 850, y)
        wrj = _vmath(ng, "SCALE", x + 1250, y, a=rj, scale=wj, parent=fr_tap)
        acc_v_n = _vmath(ng, "ADD", x + 1450, y, a=acc_v, b=wrj.outputs["Vector"], parent=fr_tap)
        acc_w_n = _math(ng, "ADD", x + 1450, y - 150, a=acc_w, b=wj, parent=fr_tap)
        acc_v, acc_w = acc_v_n.outputs["Vector"], acc_w_n.outputs[0]

    inv_w = _math(ng, "DIVIDE", 1700, 400, a=1.0, b=acc_w, parent=fr_tap)
    avg = _vmath(ng, "SCALE", 1900, 400, a=acc_v, scale=inv_w.outputs[0], parent=fr_tap, label="Weighted average")

    pick = _node(ng, "GeometryNodeSwitch", 2100, 400, parent=fr_out, input_type="VECTOR", label="Smooth Taps?")
    _link(ng, gi.outputs["Smooth Taps"], _sock(pick.inputs, "Switch"))
    _link(ng, rest0, _sock(pick.inputs, "False"))
    _link(ng, avg.outputs["Vector"], _sock(pick.inputs, "True"))

    blur = _node(ng, "GeometryNodeBlurAttribute", 2300, 400, parent=fr_out, data_type="FLOAT_VECTOR")
    _link(ng, _sock(pick.outputs, "Output"), _sock(blur.inputs, "Value"))
    _link(ng, gi.outputs["Blur Iterations"], blur.inputs["Iterations"])

    store = _node(ng, "GeometryNodeStoreNamedAttribute", 2300, 0, parent=fr_out,
                  data_type="FLOAT_VECTOR", domain="POINT")
    _link(ng, gi.outputs["Geometry"], store.inputs["Geometry"])
    _link(ng, gi.outputs["UVW Attribute"], store.inputs["Name"])
    _link(ng, _sock(blur.outputs, "Value"), _sock(store.inputs, "Value"))

    # Blur feature: smooth the uvw further and store the difference sharp-minus-blurred. The
    # "UVW Blur" shader group subtracts it from any connected UVW to pull the mapping toward the
    # smoothed field (color smearing). Stored as a delta so it stays correct even if the uvw
    # attribute itself is later overwritten (e.g. Triplanar UV Attribute set to the same name).
    blur2 = _node(ng, "GeometryNodeBlurAttribute", 2550, 400, parent=fr_out, data_type="FLOAT_VECTOR",
                  label="Color blur")
    _link(ng, _sock(blur.outputs, "Value"), _sock(blur2.inputs, "Value"))
    _link(ng, gi.outputs["Color Blur Iterations"], blur2.inputs["Iterations"])
    delta = _vmath(ng, "SUBTRACT", 2750, 400, a=_sock(blur.outputs, "Value"),
                   b=_sock(blur2.outputs, "Value"), parent=fr_out, label="sharp - blurred")

    store_blur = _node(ng, "GeometryNodeStoreNamedAttribute", 2750, 0, parent=fr_out,
                       data_type="FLOAT_VECTOR", domain="POINT", label="Store %s" % BLUR_ATTR)
    store_blur.inputs["Name"].default_value = BLUR_ATTR
    _link(ng, store.outputs["Geometry"], store_blur.inputs["Geometry"])
    _link(ng, delta.outputs["Vector"], _sock(store_blur.inputs, "Value"))
    uvw_geo = store_blur.outputs["Geometry"]

    # ---- projected UV states: each state mesh's hand-made UV is sampled at the nearest surface point to
    # the vertex's rest position (rest space -> world -> surface local) and stored as uvw_state<i>.
    fr_st = _frame(ng, "Projected UV states (UVs of the State meshes, sampled in rest space)", (0.2, 0.4, 0.4))
    adv_uvw = _sock(blur.outputs, "Value")
    self_inv = _node(ng, "FunctionNodeInvertMatrix", 2300, -1500, parent=fr_st, label="World -> surface")
    _link(ng, self_info.outputs["Transform"], self_inv.inputs["Matrix"])
    raw_back = _node(ng, "FunctionNodeMatrixMultiply", 2500, -1500, parent=fr_st, label="Rest -> surface")
    _link(ng, self_inv.outputs["Matrix"], raw_back.inputs[0])
    _link(ng, _sock(src_mat.outputs, "Output"), raw_back.inputs[1])
    cube_back = _node(ng, "FunctionNodeMatrixMultiply", 2500, -1650, parent=fr_st, label="Cube -> surface")
    _link(ng, self_inv.outputs["Matrix"], cube_back.inputs[0])
    _link(ng, cube_info.outputs["Transform"], cube_back.inputs[1])
    back_mat = _node(ng, "GeometryNodeSwitch", 2700, -1550, parent=fr_st, input_type="MATRIX",
                     label="Cube Projection?")
    _link(ng, gi.outputs["Cube Projection"], _sock(back_mat.inputs, "Switch"))
    _link(ng, raw_back.outputs[0], _sock(back_mat.inputs, "False"))
    _link(ng, cube_back.outputs[0], _sock(back_mat.inputs, "True"))
    cube_p = _vmath(ng, "MULTIPLY_ADD", 2500, -1800, a=adv_uvw, b=(2.0, 2.0, 2.0), parent=fr_st,
                    label="0..1 -> cube space")
    cube_p.inputs[2].default_value = (-1.0, -1.0, -1.0)
    back_vec = _node(ng, "GeometryNodeSwitch", 2700, -1800, parent=fr_st, input_type="VECTOR",
                     label="Cube Projection?")
    _link(ng, gi.outputs["Cube Projection"], _sock(back_vec.inputs, "Switch"))
    _link(ng, adv_uvw, _sock(back_vec.inputs, "False"))
    _link(ng, cube_p.outputs["Vector"], _sock(back_vec.inputs, "True"))
    samp_pos = _node(ng, "FunctionNodeTransformPoint", 2900, -1700, parent=fr_st,
                     label="Rest position in surface space")
    _link(ng, _sock(back_vec.outputs, "Output"), samp_pos.inputs["Vector"])
    _link(ng, _sock(back_mat.outputs, "Output"), samp_pos.inputs["Transform"])

    st_geo = uvw_geo
    for i in range(1, NUM_STATES + 1):
        y = -2000 - (i - 1) * 450
        st_info = _node(ng, "GeometryNodeObjectInfo", 2300, y, parent=fr_st, transform_space="RELATIVE",
                        label="State %d" % i)
        _link(ng, gi.outputs["State %d" % i], st_info.inputs["Object"])
        st_uv = _node(ng, "GeometryNodeInputNamedAttribute", 2300, y - 150, parent=fr_st,
                      data_type="FLOAT_VECTOR", label="State %d UV" % i)
        _link(ng, gi.outputs["State UV Map"], st_uv.inputs["Name"])
        st_samp = _node(ng, "GeometryNodeSampleNearestSurface", 2550, y, parent=fr_st,
                        data_type="FLOAT_VECTOR", label="Project state %d" % i)
        _link(ng, st_info.outputs["Geometry"], st_samp.inputs["Mesh"])
        _link(ng, _sock(st_uv.outputs, "Attribute"), _sock(st_samp.inputs, "Value"))
        _link(ng, samp_pos.outputs["Vector"], st_samp.inputs["Sample Position"])
        st_ok = _node(ng, "GeometryNodeSwitch", 2800, y, parent=fr_st, input_type="VECTOR",
                      label="State %d set? (else advected uvw)" % i)
        _link(ng, st_samp.outputs["Is Valid"], _sock(st_ok.inputs, "Switch"))
        _link(ng, adv_uvw, _sock(st_ok.inputs, "False"))
        _link(ng, _sock(st_samp.outputs, "Value"), _sock(st_ok.inputs, "True"))
        st_store = _node(ng, "GeometryNodeStoreNamedAttribute", 3050, y, parent=fr_st,
                         data_type="FLOAT_VECTOR", domain="POINT", label="Store %s" % (STATE_ATTR % i))
        st_store.inputs["Name"].default_value = STATE_ATTR % i
        _link(ng, st_geo, st_store.inputs["Geometry"])
        _link(ng, _sock(st_ok.outputs, "Output"), _sock(st_store.inputs, "Value"))
        st_geo = st_store.outputs["Geometry"]
    uvw_geo = st_geo

    # ---- triplanar projection: per-face dominant rest-space axis -> 2D UV map (face corner domain)
    fr_tri = _frame(ng, "Triplanar UVW projection (rest-space normal picks the projection plane)",
                    (0.35, 0.25, 0.4))

    uvw_in = _node(ng, "GeometryNodeInputNamedAttribute", 2300, -700, parent=fr_tri, data_type="FLOAT_VECTOR",
                   label="uvw (stored)")
    _link(ng, gi.outputs["UVW Attribute"], uvw_in.inputs["Name"])

    cap_p = _node(ng, "GeometryNodeCaptureAttribute", 2600, -300, parent=fr_tri, domain="POINT",
                  label="Keep current position")
    cap_p.capture_items.new("VECTOR", "Position")
    _link(ng, uvw_geo, cap_p.inputs["Geometry"])
    _link(ng, pos.outputs["Position"], cap_p.inputs["Position"])

    to_rest = _node(ng, "GeometryNodeSetPosition", 2850, -300, parent=fr_tri, label="Move to rest space")
    _link(ng, cap_p.outputs["Geometry"], to_rest.inputs["Geometry"])
    _link(ng, _sock(uvw_in.outputs, "Attribute"), to_rest.inputs["Position"])

    nrm = _node(ng, "GeometryNodeInputNormal", 2850, -550, parent=fr_tri)
    cap_n = _node(ng, "GeometryNodeCaptureAttribute", 3100, -300, parent=fr_tri, domain="FACE",
                  label="Rest-space face normal")
    cap_n.capture_items.new("VECTOR", "Rest Normal")
    _link(ng, to_rest.outputs["Geometry"], cap_n.inputs["Geometry"])
    _link(ng, nrm.outputs["Normal"], cap_n.inputs["Rest Normal"])

    from_rest = _node(ng, "GeometryNodeSetPosition", 3350, -300, parent=fr_tri, label="Back to current position")
    _link(ng, cap_n.outputs["Geometry"], from_rest.inputs["Geometry"])
    _link(ng, cap_p.outputs["Position"], from_rest.inputs["Position"])

    n_corner = _node(ng, "GeometryNodeFieldOnDomain", 3350, -600, parent=fr_tri, domain="CORNER",
                     data_type="FLOAT_VECTOR", label="Normal at corner")
    _link(ng, cap_n.outputs["Rest Normal"], _sock(n_corner.inputs, "Value"))
    n_abs = _vmath(ng, "ABSOLUTE", 3550, -600, a=_sock(n_corner.outputs, "Value"), parent=fr_tri)
    n_xyz = _node(ng, "ShaderNodeSeparateXYZ", 3750, -600, parent=fr_tri, label="|normal|")
    _link(ng, n_abs.outputs["Vector"], n_xyz.inputs["Vector"])
    ax, ay, az = n_xyz.outputs["X"], n_xyz.outputs["Y"], n_xyz.outputs["Z"]

    def cmp_ge(a, b, x, y):
        c = _node(ng, "FunctionNodeCompare", x, y, parent=fr_tri, data_type="FLOAT", operation="GREATER_EQUAL")
        _link(ng, a, _sock(c.inputs, "A"))
        _link(ng, b, _sock(c.inputs, "B"))
        return c.outputs["Result"]

    def cmp_gt(a, b, x, y):
        c = _node(ng, "FunctionNodeCompare", x, y, parent=fr_tri, data_type="FLOAT", operation="GREATER_THAN")
        _link(ng, a, _sock(c.inputs, "A"))
        _link(ng, b, _sock(c.inputs, "B"))
        return c.outputs["Result"]

    def b_and(a, b, x, y, label):
        n = _node(ng, "FunctionNodeBooleanMath", x, y, parent=fr_tri, operation="AND", label=label)
        _link(ng, a, n.inputs[0])
        _link(ng, b, n.inputs[1])
        return n.outputs["Boolean"]

    z_dom = b_and(cmp_ge(az, ax, 3950, -450), cmp_ge(az, ay, 3950, -600), 4150, -520, "Z dominant")
    x_dom = b_and(cmp_ge(ax, ay, 3950, -750), cmp_gt(ax, az, 3950, -900), 4150, -820, "X dominant")

    p_scaled = _vmath(ng, "SCALE", 3750, -800, a=_sock(uvw_in.outputs, "Attribute"),
                      scale=gi.outputs["Triplanar Scale"], parent=fr_tri, label="Scaled UVW")
    p_xyz = _node(ng, "ShaderNodeSeparateXYZ", 3950, -1050, parent=fr_tri, label="UVW xyz")
    _link(ng, p_scaled.outputs["Vector"], p_xyz.inputs["Vector"])

    def pick_f(cond, f, t, x, y, label):
        s = _node(ng, "GeometryNodeSwitch", x, y, parent=fr_tri, input_type="FLOAT", label=label)
        _link(ng, cond, _sock(s.inputs, "Switch"))
        _link(ng, f, _sock(s.inputs, "False"))
        _link(ng, t, _sock(s.inputs, "True"))
        return _sock(s.outputs, "Output")

    u = pick_f(x_dom, p_xyz.outputs["X"], p_xyz.outputs["Y"], 4350, -750, "U: Y if X-dominant else X")
    v = pick_f(z_dom, p_xyz.outputs["Z"], p_xyz.outputs["Y"], 4350, -950, "V: Y if Z-dominant else Z")
    uv = _node(ng, "ShaderNodeCombineXYZ", 4550, -850, parent=fr_tri, label="UV")
    _link(ng, u, uv.inputs["X"])
    _link(ng, v, uv.inputs["Y"])

    store_uv = _node(ng, "GeometryNodeStoreNamedAttribute", 4550, -300, parent=fr_tri,
                     data_type="FLOAT2", domain="CORNER", label="Store triplanar UV map")
    _link(ng, from_rest.outputs["Geometry"], store_uv.inputs["Geometry"])
    _link(ng, gi.outputs["Triplanar UV Attribute"], store_uv.inputs["Name"])
    _link(ng, uv.outputs["Vector"], _sock(store_uv.inputs, "Value"))

    tri_sw = _node(ng, "GeometryNodeSwitch", 4800, 0, parent=fr_tri, input_type="GEOMETRY", label="Triplanar?")
    _link(ng, gi.outputs["Triplanar"], _sock(tri_sw.inputs, "Switch"))
    _link(ng, uvw_geo, _sock(tri_sw.inputs, "False"))
    _link(ng, store_uv.outputs["Geometry"], _sock(tri_sw.inputs, "True"))
    _link(ng, _sock(tri_sw.outputs, "Output"), go.inputs["Geometry"])

    go.location = (5100, 0)
    gi.location = (-1400, 0)
    return ng


# --------------------------------------------------------------------------- shader nodes
def build_shader_group():
    ng = _group(SHADER_NAME, "ShaderNodeTree")

    _socket(ng, "UVW", "OUTPUT", "NodeSocketVector",
            description="Advected UVW coordinate. Plug into the Vector input of any texture")
    _socket(ng, "Island ID", "OUTPUT", "NodeSocketFloat",
            description="Random 0-1 value per UVW island (0 when islands are off)")
    _socket(ng, "Island Color", "OUTPUT", "NodeSocketColor",
            description="Random color per UVW island (black when islands are off)")
    _socket(ng, "Triplanar UV", "OUTPUT", "NodeSocketVector",
            description="2D triplanar UV from the '%s' map written by the Geometry Nodes group "
                        "(needs its Triplanar option on). Plug into a texture with Flat projection"
                        % TRIPLANAR_ATTR)

    _socket(ng, "Scale", "INPUT", "NodeSocketFloat", 1.0,
            description="Texture scale (multiplies the rest-space coordinates)")
    _socket(ng, "Random Islands", "INPUT", "NodeSocketBool", False,
            description="Break the texture into randomly offset/rotated islands")
    _socket(ng, "Island Size", "INPUT", "NodeSocketFloat", 1.0, min_value=0.001,
            description="Island size in scaled UVW units (1 = one texture tile when Scale is 1)")
    _socket(ng, "Random Rotation", "INPUT", "NodeSocketVector", (0.0, 0.0, 0.0),
            min_value=0.0, max_value=1.0, subtype="XYZ",
            description="Per-island random rotation amount per axis (0-1 = up to 360 degrees). "
                        "Use Z for flat image projection")
    _socket(ng, "Seed", "INPUT", "NodeSocketFloat", 0.0,
            description="Changes the random island layout")

    gi = _node(ng, "NodeGroupInput", -1400, 0)
    go = _node(ng, "NodeGroupOutput", 1500, 0)

    attr = _node(ng, "ShaderNodeAttribute", -1200, 250, label="uvw attribute",
                 attribute_type="GEOMETRY", attribute_name=UVW_ATTR)

    p = _vmath(ng, "SCALE", -950, 250, a=attr.outputs["Vector"], scale=gi.outputs["Scale"], label="Scaled UVW")

    size = _math(ng, "MAXIMUM", -950, 0, a=gi.outputs["Island Size"], b=1e-4, label="Size")
    inv_size = _math(ng, "DIVIDE", -750, 0, a=1.0, b=size.outputs[0])
    cell_f = _vmath(ng, "SCALE", -550, 100, a=p.outputs["Vector"], scale=inv_size.outputs[0])
    cell = _vmath(ng, "FLOOR", -350, 100, a=cell_f.outputs["Vector"], label="Island cell")
    center_c = _vmath(ng, "ADD", -350, -100, a=cell.outputs["Vector"], b=(0.5, 0.5, 0.5))
    center = _vmath(ng, "SCALE", -150, -100, a=center_c.outputs["Vector"], scale=size.outputs[0],
                    label="Island center")

    seed_b = _math(ng, "ADD", -550, -300, a=gi.outputs["Seed"], b=37.17)
    noise_a = _node(ng, "ShaderNodeTexWhiteNoise", -150, 350, label="Random offset", noise_dimensions="4D")
    _link(ng, cell.outputs["Vector"], noise_a.inputs["Vector"])
    _link(ng, gi.outputs["Seed"], noise_a.inputs["W"])
    noise_b = _node(ng, "ShaderNodeTexWhiteNoise", -150, -350, label="Random rotation", noise_dimensions="4D")
    _link(ng, cell.outputs["Vector"], noise_b.inputs["Vector"])
    _link(ng, seed_b.outputs[0], noise_b.inputs["W"])

    full_turn = _math(ng, "MULTIPLY", 50, -450, a=gi.outputs["Random Islands"], b=2.0 * math.pi,
                      label="Toggle x 2pi")
    rot_amt = _vmath(ng, "MULTIPLY", 50, -250, a=noise_b.outputs["Color"], b=gi.outputs["Random Rotation"])
    euler = _vmath(ng, "SCALE", 250, -250, a=rot_amt.outputs["Vector"], scale=full_turn.outputs[0],
                   label="Euler angles")

    rotate = _node(ng, "ShaderNodeVectorRotate", 500, 100, rotation_type="EULER_XYZ")
    _link(ng, p.outputs["Vector"], rotate.inputs["Vector"])
    _link(ng, center.outputs["Vector"], rotate.inputs["Center"])
    _link(ng, euler.outputs["Vector"], rotate.inputs["Rotation"])

    off_amt = _math(ng, "MULTIPLY", 250, 450, a=gi.outputs["Random Islands"], b=10.0, label="Toggle x offset range")
    offset = _vmath(ng, "SCALE", 500, 350, a=noise_a.outputs["Color"], scale=off_amt.outputs[0])
    final = _vmath(ng, "ADD", 800, 250, a=rotate.outputs["Vector"], b=offset.outputs["Vector"], label="UVW")
    _link(ng, final.outputs["Vector"], go.inputs["UVW"])

    iid = _math(ng, "MULTIPLY", 500, 600, a=noise_a.outputs["Value"], b=gi.outputs["Random Islands"])
    _link(ng, iid.outputs[0], go.inputs["Island ID"])
    icol = _vmath(ng, "SCALE", 500, 500, a=noise_a.outputs["Color"], scale=gi.outputs["Random Islands"])
    _link(ng, icol.outputs["Vector"], go.inputs["Island Color"])

    tri_uv = _node(ng, "ShaderNodeUVMap", 500, 750, label="triplanar UV map", uv_map=TRIPLANAR_ATTR)
    _link(ng, tri_uv.outputs["UV"], go.inputs["Triplanar UV"])

    go.location = (1100, 250)
    return ng


def build_blur_shader_group():
    """Blur feature: subtractive mix of the sharp and blurred uvw maps.

    One UVW slot - connect any source (UVW Advect Shader, Triplanar UV, a Mapping node, ...).
    Output = UVW - Blur * delta, where delta is the 'uvw_blur_delta' attribute (sharp uvw minus
    blurred uvw) written by the Geometry Nodes group. The connected mapping is pulled toward the
    smoothed field, which smears/mixes the texture colors. With the plain advected UVW connected
    and Blur = 1 you get exactly the fully blurred mapping.
    """
    ng = _group(BLUR_NAME, "ShaderNodeTree")

    _socket(ng, "UVW", "OUTPUT", "NodeSocketVector",
            description="Blurred UVW. Plug into the Vector input of any texture")

    _socket(ng, "UVW", "INPUT", "NodeSocketVector", hide_value=True,
            description="UVW map to blur - connect any source (UVW Advect Shader, Triplanar UV, "
                        "a Mapping node, ...)")
    _socket(ng, "Blur", "INPUT", "NodeSocketFloat", 1.0, min_value=0.0, max_value=100.0,
            description="Blur amount: how strongly the map is pulled toward the smoothed uvw "
                        "(0 = off, 1 = fully blurred mapping, above 1 exaggerates the effect). "
                        "Smoothing distance comes from 'Color Blur Iterations' on the %s modifier" % GN_NAME)

    gi = _node(ng, "NodeGroupInput", -800, 0)
    go = _node(ng, "NodeGroupOutput", 400, 0)

    delta = _node(ng, "ShaderNodeAttribute", -800, 250, label="sharp - blurred uvw",
                  attribute_type="GEOMETRY", attribute_name=BLUR_ATTR)
    amount = _vmath(ng, "SCALE", -400, 200, a=delta.outputs["Vector"], scale=gi.outputs["Blur"])
    out = _vmath(ng, "SUBTRACT", 0, 100, a=gi.outputs["UVW"], b=amount.outputs["Vector"],
                 label="UVW - Blur * delta")
    _link(ng, out.outputs["Vector"], go.inputs["UVW"])
    return ng


def build_states_shader_group():
    """Projected states + morph: blend the UVs projected from the State meshes.

    Morph 0 = state 1, 1 = state 2, 2 = state 3, in between = linear blend of the two neighbours.
    "UVW" is the blended UV (one texture slides between the states' mappings). "UVW 1..3" and
    "Weight 1..3" give each state's own UV and its cross-fade weight (weights always sum to 1) for
    one texture per state: color = sum(texture_i(UVW i) * Weight i).
    """
    ng = _group(STATES_NAME, "ShaderNodeTree")

    _socket(ng, "UVW", "OUTPUT", "NodeSocketVector",
            description="UV blended between the states by Morph. Plug into the Vector input "
                        "of a texture (Flat projection) or into the UVW slot of UVW Blur")
    for i in range(1, NUM_STATES + 1):
        _socket(ng, "UVW %d" % i, "OUTPUT", "NodeSocketVector",
                description="UV projected from the State %d mesh" % i)
    for i in range(1, NUM_STATES + 1):
        _socket(ng, "Weight %d" % i, "OUTPUT", "NodeSocketFloat",
                description="Cross-fade weight of state %d (1 at Morph = %d, 0 two or more steps away)"
                            % (i, i - 1))

    _socket(ng, "Morph", "INPUT", "NodeSocketFloat", 0.0, min_value=0.0, max_value=float(NUM_STATES - 1),
            description="0 = state 1, 1 = state 2, 2 = state 3. Animate it to morph between the states")

    gi = _node(ng, "NodeGroupInput", -900, 0)
    go = _node(ng, "NodeGroupOutput", 900, 0)

    lo = _math(ng, "MAXIMUM", -700, 300, a=gi.outputs["Morph"], b=0.0, label="Morph >= 0")
    m = _math(ng, "MINIMUM", -500, 300, a=lo.outputs[0], b=float(NUM_STATES - 1), label="Morph <= last state")

    acc = None
    for i in range(1, NUM_STATES + 1):
        y = 100 - (i - 1) * 350
        dist = _math(ng, "SUBTRACT", -300, y, a=m.outputs[0], b=float(i - 1))
        adist = _math(ng, "ABSOLUTE", -150, y, a=dist.outputs[0])
        w = _math(ng, "SUBTRACT", 0, y, a=1.0, b=adist.outputs[0], label="Weight %d" % i)
        w.use_clamp = True
        _link(ng, w.outputs[0], go.inputs["Weight %d" % i])

        attr = _node(ng, "ShaderNodeAttribute", -300, y - 120, label="uvw state %d" % i,
                     attribute_type="GEOMETRY", attribute_name=STATE_ATTR % i)
        _link(ng, attr.outputs["Vector"], go.inputs["UVW %d" % i])
        part = _vmath(ng, "SCALE", 250, y - 60, a=attr.outputs["Vector"], scale=w.outputs[0])
        if acc is None:
            acc = part.outputs["Vector"]
        else:
            add = _vmath(ng, "ADD", 450, y - 60, a=acc, b=part.outputs["Vector"])
            acc = add.outputs["Vector"]
    _link(ng, acc, go.inputs["UVW"])
    return ng


def build_all():
    names = (SHADER_NAME, BLUR_NAME, STATES_NAME)
    snaps = [_snapshot_instances(n) for n in names]
    mod_snap = _snapshot_modifiers()
    groups = (build_geometry_nodes(), build_shader_group(), build_blur_shader_group(),
              build_states_shader_group())
    for snap in snaps:
        _restore_instances(snap)
    _restore_modifiers(mod_snap)
    return groups


def _snapshot_modifiers():
    """Inputs (by socket name) of every modifier using the GN group; rebuilding resets them."""
    snap = []
    gn = bpy.data.node_groups.get(GN_NAME)
    if gn is None:
        return snap
    for ob in bpy.data.objects:
        for mod in ob.modifiers:
            if mod.type == "NODES" and mod.node_group == gn:
                vals = {}
                for it in gn.interface.items_tree:
                    if it.item_type == "SOCKET" and it.in_out == "INPUT" and it.name != "Geometry":
                        try:
                            vals[it.name] = get_modifier_input(mod, it.name)
                        except Exception:
                            pass
                snap.append((ob.name, mod.name, vals))
    return snap


def _restore_modifiers(snap):
    gn = bpy.data.node_groups.get(GN_NAME)
    for ob_name, mod_name, vals in snap:
        mod = bpy.data.objects[ob_name].modifiers[mod_name]
        for name, val in vals.items():
            try:
                set_modifier_input(mod, name, val)
            except Exception:
                pass
        # inputs added by a newer script version start empty on old modifiers; give them their defaults
        for it in gn.interface.items_tree:
            if (it.item_type == "SOCKET" and it.in_out == "INPUT" and it.name != "Geometry"
                    and it.name not in vals and hasattr(it, "default_value")):
                try:
                    set_modifier_input(mod, it.name, it.default_value)
                except Exception:
                    pass


# --------------------------------------------------------------------------- one-click setup
def _modifier_input_id(ng, name):
    for item in ng.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name == name:
            return item.identifier
    raise KeyError(name)


def set_modifier_input(mod, name, value):
    """Set a Geometry Nodes modifier input by its display name (Blender 4.5 - 5.2+)."""
    ident = _modifier_input_id(mod.node_group, name)
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    if inputs is not None:
        getattr(inputs, ident).value = value
    else:
        mod[ident] = value


def get_modifier_input(mod, name):
    ident = _modifier_input_id(mod.node_group, name)
    inputs = getattr(getattr(mod, "properties", None), "inputs", None)
    return getattr(inputs, ident).value if inputs is not None else mod[ident]


def setup(surface, particles, image=None):
    if GN_NAME not in bpy.data.node_groups:
        gn, sh, blur, _states = build_all()
    else:
        gn = bpy.data.node_groups[GN_NAME]
        sh = bpy.data.node_groups.get(SHADER_NAME) or build_shader_group()
        blur = bpy.data.node_groups.get(BLUR_NAME) or build_blur_shader_group()
        if STATES_NAME not in bpy.data.node_groups:
            build_states_shader_group()

    mod = surface.modifiers.get(GN_NAME) or surface.modifiers.new(GN_NAME, "NODES")
    mod.node_group = gn
    set_modifier_input(mod, "Particles", particles)

    if not surface.material_slots:
        surface.data.materials.append(bpy.data.materials.new("UVW Advect Material"))
    mat = surface.material_slots[0].material
    if mat is None:
        mat = bpy.data.materials.new("UVW Advect Material")
        surface.material_slots[0].material = mat
    mat.use_nodes = True
    nt = mat.node_tree
    if not any(n.bl_idname == "ShaderNodeGroup" and n.node_tree == sh for n in nt.nodes):
        bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if bsdf is None:
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
            out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None) \
                or nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        grp = nt.nodes.new("ShaderNodeGroup")
        grp.node_tree = sh
        grp.location = (bsdf.location.x - 1000, bsdf.location.y)
        bgrp = nt.nodes.new("ShaderNodeGroup")
        bgrp.node_tree = blur
        bgrp.location = (bsdf.location.x - 700, bsdf.location.y)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.projection = "BOX"
        tex.projection_blend = 0.2
        tex.location = (bsdf.location.x - 400, bsdf.location.y)
        if image is not None:
            tex.image = image
        nt.links.new(grp.outputs["UVW"], bgrp.inputs["UVW"])
        nt.links.new(bgrp.outputs["UVW"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mod, mat


def add_projection_cube(surface, name="UVW Projection Cube"):
    """Create a cube Empty fitted to the surface's current (first frame) bounds and use it for the start UVW."""
    from mathutils import Matrix, Vector

    mod = surface.modifiers.get(GN_NAME)
    if mod is None:
        raise RuntimeError("'%s' has no '%s' modifier; run UVW Advect: Setup first" % (surface.name, GN_NAME))
    corners = [surface.matrix_world @ Vector(c) for c in surface.bound_box]
    lo = Vector((min(c[i] for c in corners) for i in range(3)))
    hi = Vector((max(c[i] for c in corners) for i in range(3)))
    half = max((hi - lo)) * 0.5 or 0.5

    cube = bpy.data.objects.get(name) or bpy.data.objects.new(name, None)
    if cube.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(cube)
    cube.empty_display_type = "CUBE"
    cube.empty_display_size = 1.0
    cube.matrix_world = Matrix.Translation((lo + hi) * 0.5) @ Matrix.Scale(half, 4)
    set_modifier_input(mod, "Projection Object", cube)
    set_modifier_input(mod, "Cube Projection", True)
    return cube


class UVWADVECT_OT_setup(bpy.types.Operator):
    """Add the UVW Advect modifier and shader group. Active = surface mesh, other selected = point cloud"""
    bl_idname = "object.uvw_advect_setup"
    bl_label = "UVW Advect: Setup"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        surface = context.active_object
        others = [o for o in context.selected_objects if o != surface]
        if not others:
            self.report({"ERROR"}, "Select the point cloud object as well (active object = surface mesh)")
            return {"CANCELLED"}
        setup(surface, others[0])
        self.report({"INFO"}, "UVW Advect set up on '%s' using particles '%s'" % (surface.name, others[0].name))
        return {"FINISHED"}


class UVWADVECT_OT_add_cube(bpy.types.Operator):
    """Create a cube Empty fitted to the active surface and use it to place the start UVW (cube projection).
Run it on the first frame, then keep the cube still"""
    bl_idname = "object.uvw_advect_add_cube"
    bl_label = "UVW Advect: Add Projection Cube"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH" and ob.modifiers.get(GN_NAME) is not None

    def execute(self, context):
        cube = add_projection_cube(context.active_object)
        self.report({"INFO"}, "Cube projection set from '%s' (move/rotate/scale it on the first frame)" % cube.name)
        return {"FINISHED"}


_classes = (UVWADVECT_OT_setup, UVWADVECT_OT_add_cube)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
    build_all()
    print("UVW Advect node groups built:", GN_NAME, "/", SHADER_NAME, "/", BLUR_NAME, "/", STATES_NAME)
