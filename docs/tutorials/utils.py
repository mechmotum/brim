"""Utilities for the tutorials."""
import inspect
import os
import zipfile

import requests
import sympy as sm
import sympy.physics.mechanics as me
from brim.bicycle import WheelBase
from brim.utilities.utilities import check_zero
from sympy.core.function import AppliedUndef

BICYCLE_PARAMETERS_ZIP_URL = (
    "https://github.com/moorepants/BicycleParameters/archive/refs/heads/master.zip")
TUTORIALS_DIR = os.path.dirname(__file__)


def download_parametrization_data(data_dir: str | None = None) -> None:
    """Download the bicycle and rider data from the BicycleParameters repository."""
    if data_dir is None:
        data_dir = os.path.join(TUTORIALS_DIR, "data")
    if os.path.exists(os.path.join(TUTORIALS_DIR, "data")):
        raise FileExistsError("The data folder already exists.")
    # Download the zip file.
    zip_file = os.path.join(os.path.dirname(__file__), "bicycle_parameters.zip")
    with open(zip_file, "wb") as f:
        f.write(requests.get(BICYCLE_PARAMETERS_ZIP_URL).content)
    # Obtain the data folder from the zip file.
    with zipfile.ZipFile(zip_file, "r") as f:
        for file in f.namelist():
            if file.startswith("BicycleParameters-master/data/"):
                f.extract(file, TUTORIALS_DIR)
    # Remove the zip file.
    os.remove(zip_file)
    # Rename the data folder.
    os.rename(os.path.join(TUTORIALS_DIR, "BicycleParameters-master/data/"), data_dir)
    # Remove the BicycleParameters-master folder.
    os.rmdir(os.path.join(TUTORIALS_DIR, "BicycleParameters-master"))


def check_documentation(*objects: object) -> None:
    """Check if type signature and a docstring are provided for each object."""
    objects = list(objects)
    errors = []
    while objects:
        obj = objects.pop()
        if isinstance(obj, property):
            obj_str = f"'{obj.fget.__name__}'"
        else:
            obj_str = f"'{obj.__name__}'"
        if not obj.__doc__:
            errors.append(f"{obj_str} has no docstring.")
        if isinstance(obj, property):
            if "return" not in obj.fget.__annotations__:
                errors.append(f"{obj_str} has no type hint for what it returns")
            continue
        if inspect.isclass(obj):
            continue
        sig = inspect.signature(obj)
        for par in sig.parameters:
            if par not in ("self",) and par not in obj.__annotations__:
                errors.append(f"'{par}' from {obj_str} does not have a type hint.")
        if "return" not in obj.__annotations__:
            errors.append(f"{obj_str} has no type hint for what it returns")
    if errors:
        raise Exception("\n".join(errors))

def verify_ground_base(ground_base_cls: type) -> None:
    """Verify GroundBase from the rolling_disc_from_scratch tutorial."""
    check_documentation(ground_base_cls.origin, ground_base_cls.tangent_vectors)

def verify_flat_ground(flat_ground_cls: type) -> None:
    """Verify FlatGround from the rolling_disc_from_scratch tutorial."""
    ground = flat_ground_cls("test")
    ground.define_all()
    assert ground.normal == -ground.frame.z
    assert ground.frame.x == ground.tangent_vectors[0]
    assert ground.frame.y == ground.tangent_vectors[1]
    a, b = sm.symbols("a b")
    point = me.Point("test")
    ground.set_pos_point(point, (a, b))
    assert point.pos_from(ground.origin) == a * ground.frame.x + b * ground.frame.y
    check_documentation(flat_ground_cls.tangent_vectors, flat_ground_cls.set_pos_point)

def verify_knife_edge_wheel(knife_edge_wheel_cls: type) -> None:
    """Verify KnifeEdgeWheel from the rolling_disc_from_scratch tutorial."""
    wheel = knife_edge_wheel_cls("test")
    assert isinstance(wheel, WheelBase)
    wheel.define_all()
    assert wheel.symbols["r"].name.startswith("test")
    assert wheel.descriptions[wheel.symbols["r"]]
    assert wheel.center == wheel.body.masscenter
    assert wheel.rotation_axis == wheel.frame.y
    check_documentation(knife_edge_wheel_cls.center, knife_edge_wheel_cls.rotation_axis)

def verify_tire_base(tire_base_cls: type, flat_ground_cls: type,
                     knife_edge_wheel_cls: type) -> None:
    """Verify TireBase from the rolling_disc_from_scratch tutorial."""
    tire = tire_base_cls("tire")
    try:
        tire.wheel = flat_ground_cls("ground")
    except TypeError:
        pass
    else:
        raise ValueError("The wheel property was not set correctly, "
                         "as FlatGround could be set as wheel.")
    try:
        tire.ground = knife_edge_wheel_cls("wheel")
    except TypeError:
        pass
    else:
        raise ValueError("The ground property was not set correctly, "
                         "as KnifeEdgeWheel could be set as ground.")
    tire.wheel = knife_edge_wheel_cls("wheel")
    tire.ground = flat_ground_cls("ground")
    tire.ground.define_objects()
    tire.wheel.define_objects()
    tire.define_objects()
    assert isinstance(tire.system, me.System)
    assert isinstance(tire.contact_point, me.Point)
    assert tire.on_ground is False
    tire.on_ground = True
    assert tire.on_ground is True

def verify_rolling_disc(rolling_disc_cls: type, flat_ground_cls: type,
                        knife_edge_wheel_cls: type, nonholonomic_tire_cls: type
                       ) -> None:
    """Verify RollingDisc from the rolling_disc_from_scratch tutorial."""
    rolling_disc = rolling_disc_cls("sys")
    if len(rolling_disc.required_models) != 2:
        raise AssertionError("Expected the RollingDisc to have 2 submodels.")
    if len(rolling_disc.required_connections) != 1:
        raise AssertionError("Expected the RollingDisc to have 1 connections.")
    for prop_name in ("wheel", "ground", "tire"):
        if not hasattr(rolling_disc, prop_name):
            raise AssertionError(f"RollingDisc has no {prop_name!r} property.")
    rolling_disc.wheel = knife_edge_wheel_cls("wheel")
    rolling_disc.ground = flat_ground_cls("ground")
    rolling_disc.tire = nonholonomic_tire_cls("tire")
    rolling_disc.define_connections()
    assert rolling_disc.tire.ground == rolling_disc.ground
    assert rolling_disc.tire.wheel == rolling_disc.wheel
    rolling_disc.define_objects()
    if "r" not in rolling_disc.wheel.symbols:
        raise AssertionError("You have broken the traversal of define_objects.")
    assert len(rolling_disc.q) == 5
    assert len(rolling_disc.u) == 5
    assert isinstance(rolling_disc.q, sm.MutableMatrix)
    assert isinstance(rolling_disc.u, sm.MutableMatrix)
    assert rolling_disc.tire.on_ground is True
    assert isinstance(rolling_disc.system, me.System)
    for sym in (*(rolling_disc.q), *(rolling_disc.u)):
        if not rolling_disc.descriptions.get(sym):
            raise AssertionError(f"{sym!r} has no valid description.")
    rolling_disc.define_kinematics()
    try:
        rolling_disc.system.frame.dcm(rolling_disc.ground.frame)
    except ValueError as e:
        raise AssertionError("The frame of the ground is not defined w.r.t. "
                             "`rolling_disc.system.frame`.") from e
    g_f, w_f = me.ReferenceFrame("g_f"), me.ReferenceFrame("w_f")
    w_f.orient_body_fixed(g_f, sm.symbols("q3 q4 q5", cls=sm.Wild), "zxy")
    res = rolling_disc.wheel.frame.dcm(rolling_disc.ground.frame).match(w_f.dcm(g_f))
    if len(res) != 3:
        raise AssertionError("The wheel does not seem to have been oriented correctly.")
    assert rolling_disc.tire.contact_point.pos_from(rolling_disc.ground.origin)
    assert rolling_disc.tire.contact_point.pos_from(
        rolling_disc.ground.origin).dot(rolling_disc.ground.normal) == 0
    assert rolling_disc.wheel.center.pos_from(rolling_disc.ground.origin)
    assert len(rolling_disc.system.q) == 5
    assert len(rolling_disc.system.u) == 5
    assert len(rolling_disc.system.kdes) == 5
    rolling_disc.define_loads()
    rolling_disc.define_constraints()
    assert len(rolling_disc.tire.system.holonomic_constraints) == 0
    assert len(rolling_disc.tire.system.nonholonomic_constraints) == 2
    for constr in rolling_disc.tire.system.nonholonomic_constraints:
        assert not check_zero(constr)

def verify_rolling_disc_control(
        rolling_disc_control_cls: type, rolling_disc_cls: type,
        flat_ground_cls: type, knife_edge_wheel_cls: type,
        nonholonomic_tire_cls: type
) -> None:
    """Verify RollingDiscControl from the rolling_disc_from_scratch tutorial."""
    assert rolling_disc_control_cls.required_parent_type is rolling_disc_cls
    rolling_disc = rolling_disc_cls("sys")
    rolling_disc.wheel = knife_edge_wheel_cls("wheel")
    rolling_disc.ground = flat_ground_cls("ground")
    rolling_disc.tire = nonholonomic_tire_cls("tire")
    lg = rolling_disc_control_cls("control")
    rolling_disc.add_load_groups(lg)
    rolling_disc.define_connections()
    rolling_disc.define_objects()
    for key in ("T_drive", "T_roll", "T_steer"):
        if key not in lg.symbols:
            raise AssertionError(f"No symbol under the name {key!r} found.")
        if not lg.get_description(lg.symbols[key]):
            raise AssertionError(f"{lg.symbols[key]!r} has no valid description")
        if not lg.symbols[key].name.startswith("control"):
            raise AssertionError(
                "You should use `self._add_prefix` when naming your symbols.")
        if not (isinstance(lg.symbols[key], AppliedUndef) and
                lg.symbols[key].args == (me.dynamicsymbols._t,)):
            raise AssertionError(f"{lg.symbols[key]!r} is not a dynamicsymbol.")
    rolling_disc.define_kinematics()
    rolling_disc.define_loads()
    assert lg.system.loads
    tot = me.Vector(0)
    for ld in lg.system.loads:
        assert isinstance(ld, me.Torque)
        assert ld.frame.ang_vel_in(rolling_disc.wheel.frame) == 0
        tot += ld.torque
    assert {lg.symbols[k] for k in ("T_drive", "T_roll", "T_steer")}.issubset(
        me.find_dynamicsymbols(tot, reference_frame=rolling_disc.system.frame))
