"""Module containing the models of the front frame of a bicycle."""
from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import TYPE_CHECKING

from sympy import MutableMatrix, Symbol, symbols
from sympy.physics.mechanics import (
    Force,
    Point,
    RigidBody,
    System,
    dynamicsymbols,
    inertia,
)

from symbrim.core import Attachment, Hub, ModelBase, set_default_convention

with contextlib.suppress(ImportError):
    import numpy as np
    from bicycleparameters.io import remove_uncertainties
    from dtk.bicycle import benchmark_to_moore
    from scipy.optimize import fsolve

    from symbrim.utilities.parametrize import get_inertia_vals

    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel

__all__ = ["FrontFrameBase", "RigidFrontFrame", "RigidFrontFrameMoore",
           "SuspensionRigidFrontFrame", "SuspensionRigidFrontFrameMoore"]


class FrontFrameBase(ModelBase):
    """Base class for the front frame of a bicycle.

    Explanation
    -----------
    This class in a abstract base class for the front frame of a bicycle. It defines
    the common attributes of the front frame and the methods that must be implemented
    by the subclasses.
    """

    @property
    @abstractmethod
    def steer_hub(self) -> Hub:
        """Steer hub expressed in the front frame."""

    @property
    @abstractmethod
    def wheel_hub(self) -> Hub:
        """Wheel hub expressed in the front frame."""

    @property
    @abstractmethod
    def left_hand_grip(self) -> Attachment:
        """Attachment representing the left hand grip."""

    @property
    @abstractmethod
    def right_hand_grip(self) -> Attachment:
        """Attachment representing the right hand grip."""

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        steer_top = self.steer_hub.point.locatenew(
            "P", self.steer_hub.axis * self.left_hand_grip.point.pos_from(
                self.steer_hub.point).dot(self.steer_hub.axis))
        plot_object.add_line([
            self.wheel_hub.point, self.steer_hub.point, steer_top,
            self.left_hand_grip.point, steer_top, self.right_hand_grip.point],
            self.name)
        for body in self.system.bodies:
            plot_object.add_body(body)


@set_default_convention("moore")
class RigidFrontFrame(FrontFrameBase):
    """Rigid front frame."""

    def _define_objects(self) -> None:
        """Define the objects of the front frame."""
        super()._define_objects()
        self._body = RigidBody(self._add_prefix("body"))
        self.body.central_inertia = inertia(self.body.frame,
                                            *symbols(self._add_prefix("ixx iyy izz")),
                                            izx=Symbol(self._add_prefix("izx")))
        self._system = System.from_newtonian(self.body)
        self._left_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("left_hand_grip")))
        self._right_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("right_hand_grip")))

    def _define_kinematics(self) -> None:
        """Define the kinematics of the front frame."""
        self.body.masscenter.set_vel(self.body.frame, 0)
        self.wheel_hub.point.set_vel(self.body.frame, 0)
        self.steer_hub.point.set_vel(self.body.frame, 0)
        self.left_hand_grip.point.set_vel(self.body.frame, 0)
        self.right_hand_grip.point.set_vel(self.body.frame, 0)

    @property
    def body(self) -> RigidBody:
        """Rigid body representing the front frame."""
        return self._body

    @property
    def left_hand_grip(self) -> Attachment:
        """Attachment representing the left hand grip."""
        return self._left_hand_grip

    @property
    def right_hand_grip(self) -> Attachment:
        """Attachment representing the right hand grip."""
        return self._right_hand_grip

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the front frame."""
        params = super().get_param_values(bicycle_parameters)
        str_params = _get_front_frame_moore_params(bicycle_parameters)
        if "mass" in str_params:
            params[self.body.mass] = str_params["mass"]
        if "inertia_vals" in str_params:
            params.update(get_inertia_vals(self.body, *str_params["inertia_vals"]))
        return params


class RigidFrontFrameMoore(RigidFrontFrame):
    """Rigid front frame model based on Moore's convention."""

    convention: str = "moore"

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the attributes of the front frame."""
        return {
            **super().descriptions,
            self.symbols["d2"]: "Distance between wheels along the steer axis.",
            self.symbols["d3"]: "Perpendicular distance from the steer axis to the "
                                "center of the front wheel (fork offset).",
            self.symbols["l3"]: "Distance in the front frame x direction from the front"
                                " wheel center to the center of mass of the front "
                                "frame.",
            self.symbols["l4"]: "Distance in the front frame z direction from the front"
                                " wheel center to the center of mass of the front "
                                "frame.",
            self.symbols["d6"]: "Perpendicular distance from the steer axis to the "
                                "hand grips. The hand grips are in front of the steer "
                                "axis if d6 is positive.",
            self.symbols["d7"]: "Half of the distance between the hand grips.",
            self.symbols["d8"]: "Distance of the hand grips from the steer point along "
                                "the steer axis. The hand grips are below the steer "
                                "attachment if d8 is positive.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the front frame."""
        super()._define_objects()
        self._left_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("left_hand_grip")))
        self._right_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("right_hand_grip")))
        self.symbols.update({
            name: Symbol(self._add_prefix(name)) for name in (
                "d2", "d3", "l3", "l4", "d6", "d7", "d8")})
        self._wheel_hub = Hub(self.body.frame,
                              Point(self._add_prefix("wheel_hub_point")), "y")
        self._steer_hub = Hub(self.body.frame,
                              Point(self._add_prefix("steer_hub_point")), "z")

    def _define_kinematics(self) -> None:
        """Define the kinematics of the front frame."""
        super()._define_kinematics()
        d2, d3, l3, l4, d6, d7, d8 = (self.symbols[name] for name in (
            "d2", "d3", "l3", "l4", "d6", "d7", "d8"))
        x, y, z = self.body.x, self.body.y, self.body.z
        self.wheel_hub.point.set_pos(self.steer_hub.point, d3 * x + d2 * z)
        self.body.masscenter.set_pos(self.wheel_hub.point, l3 * x + l4 * z)
        self.left_hand_grip.point.set_pos(
            self.steer_hub.point, d6 * x - d7 * y + d8 * z)
        self.right_hand_grip.point.set_pos(
            self.steer_hub.point, d6 * x + d7 * y + d8 * z)
        self.steer_hub.point.set_vel(self.body.frame, 0)

    @property
    def steer_hub(self) -> Hub:
        """Steer axis expressed in the front frame."""
        return self._steer_hub

    @property
    def wheel_hub(self) -> Hub:
        """Wheel axis expressed in the front frame."""
        return self._wheel_hub

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the front frame."""
        params = super().get_param_values(bicycle_parameters)
        str_params = _get_front_frame_moore_params(bicycle_parameters)
        for name in ("d2", "d3", "l3", "l4", "d6", "d7", "d8"):
            if name in str_params:
                params[self.symbols[name]] = str_params[name]
        return params


@set_default_convention("moore")
class SuspensionRigidFrontFrame(FrontFrameBase):
    """Front frame with suspension and no structural flexibility."""

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the attributes of the front frame."""
        return {
            **super().descriptions,
            self.q[0]: "Compression of the front suspension.",
            self.u[0]: "Compression velocity of the front suspension.",
            self.symbols["k"]: "Spring stiffness of the front suspension.",
            self.symbols["c"]: "Damping coefficient of the front suspension.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the front frame."""
        super()._define_objects()
        self.q = MutableMatrix([dynamicsymbols(self._add_prefix("q"))])
        self.u = MutableMatrix([dynamicsymbols(self._add_prefix("u"))])
        self.symbols["k"] = Symbol(self._add_prefix("k"))
        self.symbols["c"] = Symbol(self._add_prefix("c"))


class SuspensionRigidFrontFrameMoore(SuspensionRigidFrontFrame):
    """Front frame with suspension and no structural flexibility.

    Explanation
    -----------
    The front frame is modeled as a rigid body with a suspension. The suspension is
    modeled as a spring-damper system. The forces are applied parallel to the steer
    axis with at a parametrized distance ``symbols["d9"]`` from the steer axis.
    """

    convention: str = "moore"

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the attributes of the front frame."""
        return {
            **super().descriptions,
            self.symbols["d2"]: f"Distance between wheels along the steer axis when "
                                f"{self.q[0]} is zero.",
            self.symbols["d3"]: "Perpendicular distance from the steer axis to the "
                                "center of the front wheel (fork offset).",
            self.symbols["l3"]: "Distance in the front frame x direction from the front"
                                " wheel center to the center of mass of the front "
                                "frame.",
            self.symbols["l4"]: f"Distance in the front frame z direction from the "
                                f"front wheel center to the center of mass of the front"
                                f" frame when {self.q[0]} is zero.",
            self.symbols["d6"]: "Perpendicular distance from the steer axis to the "
                                "hand grips. The hand grips are in front of the steer "
                                "axis if d6 is positive.",
            self.symbols["d7"]: "Half of the distance between the hand grips.",
            self.symbols["d8"]: "Distance of the hand grips from the steer point along "
                                "the steer axis. The hand grips are below the steer "
                                "attachment if d8 is positive.",
            self.symbols["d9"]: "Perpendicular distance from the steer axis to the "
                                "suspension.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the front frame."""
        super()._define_objects()
        self._body = RigidBody(self._add_prefix("body"))
        self.body.central_inertia = inertia(self.body.frame,
                                            *symbols(self._add_prefix("ixx iyy izz")),
                                            izx=Symbol(self._add_prefix("izx")))
        self._system = System.from_newtonian(self.body)
        self.symbols.update({
            name: Symbol(self._add_prefix(name)) for name in (
                "d2", "d3", "l3", "l4", "d6", "d7", "d8", "d9")})
        self._wheel_hub = Hub.from_name(self._add_prefix("wheel"), "y")
        self._steer_hub = Hub(
            self.body.frame, Point(self._add_prefix("steer_hub_point")), "z")
        self._left_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("left_hand_grip")))
        self._right_hand_grip = Attachment(
            self.body.frame, Point(self._add_prefix("right_hand_grip")))
        self._suspension_stanchions = Point(self._add_prefix("suspension_stanchions"))
        self._suspension_lowers = Point(self._add_prefix("suspension_lowers"))

    def _define_kinematics(self) -> None:
        """Define the kinematics of the front frame."""
        super()._define_kinematics()
        d2, d3, l3, l4, d6, d7, d8, d9 = (self.symbols[name] for name in (
            "d2", "d3", "l3", "l4", "d6", "d7", "d8", "d9"))
        x, y, z = self.body.x, self.body.y, self.body.z
        self.wheel_hub.frame.orient_axis(self.body.frame, self.body.z, 0)
        self.wheel_hub.point.set_pos(
            self.steer_hub.point, d3 * x + (d2 - self.q[0]) * z)
        self.body.masscenter.set_pos(
            self.wheel_hub.point, l3 * x + (l4 + self.q[0]) * z)
        self.left_hand_grip.point.set_pos(
            self.steer_hub.point, d6 * x - d7 * y + d8 * z)
        self.right_hand_grip.point.set_pos(
            self.steer_hub.point, d6 * x + d7 * y + d8 * z)
        self.suspension_stanchions.set_pos(self.steer_hub.point, d9 * x)
        self.suspension_lowers.set_pos(self.suspension_stanchions, -self.q[0] * z)
        self.body.masscenter.set_vel(self.body.frame, 0)
        self.wheel_hub.point.set_vel(self.body.frame, -self.u[0] * z)
        self.suspension_stanchions.set_vel(self.body.frame, 0)
        self.suspension_lowers.set_vel(self.body.frame, -self.u[0] * z)
        self.system.add_coordinates(*self.q)
        self.system.add_speeds(*self.u)
        self.system.add_kdes(*(self.q.diff(dynamicsymbols._t) - self.u))

    def _define_loads(self) -> None:
        """Define the loads of the front frame."""
        super()._define_loads()
        force = -self.symbols["k"] * self.q[0] - self.symbols["c"] * self.u[0]
        self.system.add_loads(
            Force(self.suspension_stanchions, force * self.body.z),
            Force(self.suspension_lowers, -force * self.body.z)
        )

    @property
    def body(self) -> RigidBody:
        """Rigid body representing the front frame."""
        return self._body

    @property
    def left_hand_grip(self) -> Attachment:
        """Attachment representing the left hand grip."""
        return self._left_hand_grip

    @property
    def right_hand_grip(self) -> Attachment:
        """Attachment representing the right hand grip."""
        return self._right_hand_grip

    @property
    def steer_hub(self) -> Hub:
        """Steer axis expressed in the front frame."""
        return self._steer_hub

    @property
    def wheel_hub(self) -> Hub:
        """Wheel axis expressed in the front frame."""
        return self._wheel_hub

    @property
    def suspension_stanchions(self) -> Point:
        """Point representing the suspension stanchions, where the force is applied."""
        return self._suspension_stanchions

    @property
    def suspension_lowers(self) -> Point:
        """Point representing the suspension slider, where the force is applied."""
        return self._suspension_lowers

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the front frame."""
        params = super().get_param_values(bicycle_parameters)
        str_params = _get_front_frame_moore_params(bicycle_parameters)
        if "mass" in str_params:
            params[self.body.mass] = str_params["mass"]
        if "inertia_vals" in str_params:
            params.update(get_inertia_vals(self.body, *str_params["inertia_vals"]))
        for name in ("d2", "d3", "l3", "l4", "d6", "d7", "d8"):
            if name in str_params:
                params[self.symbols[name]] = str_params[name]
        return params


def _get_front_frame_moore_params(bicycle_parameters: Bicycle
                                  ) -> dict[str, object]:  # pragma: no cover
    params = {}
    if "Benchmark" in bicycle_parameters.parameters:
        bp = remove_uncertainties(bicycle_parameters.parameters["Benchmark"])
        mop = benchmark_to_moore(bp)
        params["mass"] = mop["me"]
        params["inertia_vals"] = (mop["ie11"], mop["ie22"], mop["ie33"], mop["ie12"],
                                  mop["ie23"], mop["ie31"])
        for name in ("d2", "d3", "l3", "l4"):
            params[name] = mop[name]
    if "Measured" in bicycle_parameters.parameters:
        mep = remove_uncertainties(bicycle_parameters.parameters["Measured"])
        rr, rf, w, lamht, whb, lhbr, lhbf = (mep.get(name) for name in (
            "rR", "rF", "w", "lamht", "whb", "LhbR", "LhbF"))
        if "Benchmark" in bicycle_parameters.parameters:
            rr = bp["rR"]
            rf = bp["rF"]
            w = bp["w"]
            lamht = np.pi / 2 - bp["lam"]
            d2, d3 = mop["d2"], mop["d3"]
        else:
            d2, d3 = None, None
        if whb is not None:
            params["d7"] = mep["whb"] / 2
        if not any(value is None for value in (rr, rf, w, lamht, whb, lhbr, lhbf,
                                               d2, d3)):
            def f(vals: tuple[float, ...]) -> tuple[float, ...]:
                ay, az, by, bz = vals
                return (
                    -lhbf + np.sqrt(ay ** 2 + az ** 2 + 0.25 * whb ** 2),
                    -lhbr + np.sqrt(by ** 2 + bz ** 2 + 0.25 * whb ** 2),
                    ay - by - w,
                    az - bz + rf - rr
                )

            ay, az, by, bz = fsolve(f, (0.3, 0.8, -0.7, 0.8))
            params["d6"] = -(ay * np.sin(lamht) - az * np.cos(lamht) - d3)
            params["d8"] = -(ay * np.cos(lamht) + az * np.sin(lamht) - d2)
    return params
