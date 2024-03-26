"""Module containing tire models for bicycles."""
from __future__ import annotations

from typing import Any

from sympy import Expr, atan2
from sympy import MutableDenseMatrix as Matrix
from sympy.physics.mechanics import (
    Force,
    Point,
    System,
    Torque,
    Vector,
    cross,
    dynamicsymbols,
)

from brim.bicycle.grounds import FlatGround, GroundBase
from brim.bicycle.wheels import KnifeEdgeWheel, ToroidalWheel, WheelBase
from brim.core import ConnectionBase, ModelRequirement
from brim.utilities.utilities import check_zero

__all__ = ["TireBase", "InContactTire", "NonHolonomicTire"]


class TireBase(ConnectionBase):
    """Base class for the tire model connectors."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("ground", GroundBase, "Submodel of the ground."),
        ModelRequirement("wheel", WheelBase, "Submodel of the wheel."),
    )
    ground: GroundBase
    wheel: WheelBase

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._on_ground = None

    def _set_pos_contact_point(self) -> None:
        """Compute the contact point of the wheel with the ground."""
        if isinstance(self.ground, FlatGround):
            if isinstance(self.wheel, KnifeEdgeWheel):
                self.wheel.center.set_pos(self.contact_point,
                                          self.wheel.radius * self.upward_radial_axis)
                return
            elif isinstance(self.wheel, ToroidalWheel):
                self.wheel.center.set_pos(
                    self.contact_point,
                    self.wheel.radius * self.upward_radial_axis +
                    self.wheel.transverse_radius * self.ground.get_normal(
                        self.contact_point))
                return
        raise NotImplementedError(
            f"Computation of the contact point has not been implemented for the "
            f"combination of {type(self.ground)} and {type(self.wheel)}.")

    def _define_objects(self) -> None:
        """Define the objects of the tire model."""
        super()._define_objects()
        self._system = System.from_newtonian(self.ground.body)
        self._contact_point = Point(self._add_prefix("contact_point"))
        self._upward_radial_axis = None
        self._longitudinal_axis = None
        self._lateral_axis = None

    @property
    def upward_radial_axis(self) -> Vector:
        """Wheel radial axis pointing upward from the contact point to the wheel center.

        Explanation
        -----------
        To possibly simplify the equations of motion, one can overwrite the axis used
        to compute the location of the contact point with respect to the wheel center.
        This axis should be normalized. For a knife-edge wheel, one could express the
        vector from the wheel center to the contact point of the ground as
        ``wheel.radius * upward_radial_axis``.
        """
        if self._upward_radial_axis is None:
            self._upward_radial_axis = cross(
                self.wheel.rotation_axis,
                cross(self.ground.get_normal(self.contact_point),
                      self.wheel.rotation_axis)
            ).normalize()
        return self._upward_radial_axis

    @upward_radial_axis.setter
    def upward_radial_axis(self, axis: Vector) -> None:
        name = "The upward radial axis of the wheel"
        if not isinstance(axis, Vector):
            raise TypeError(f"{name} should be a vector, but received a {type(axis)}")
        if not check_zero(axis.magnitude() - 1):
            raise ValueError(f"{name} should be normalized.")
        if not check_zero(axis.dot(self.wheel.rotation_axis)):
            raise ValueError(f"{name} should be perpendicular to the rotation axis.")
        if not check_zero(axis.dot(cross(self.ground.get_normal(self.contact_point),
                                         self.wheel.rotation_axis))):
            raise ValueError(
                f"{name} should be perpendicular to the longitudinal axis.")
        self._upward_radial_axis = axis

    @property
    def longitudinal_axis(self) -> Vector:
        """Longitudinal axis of the wheel."""
        if self._longitudinal_axis is None:
            self._longitudinal_axis = cross(self.ground.get_normal(self.contact_point),
                                            self.wheel.rotation_axis).normalize()
        return self._longitudinal_axis

    @longitudinal_axis.setter
    def longitudinal_axis(self, axis: Vector) -> None:
        name = "The longitudinal axis of the wheel"
        if not isinstance(axis, Vector):
            raise TypeError(f"{name} should be a vector, but received a {type(axis)}")
        if not check_zero(axis.magnitude() - 1):
            raise ValueError(f"{name} should be normalized.")
        if not check_zero(axis.dot(self.wheel.rotation_axis)):
            raise ValueError(f"{name} should be perpendicular to the rotation axis.")
        if not check_zero(axis.dot(self.ground.get_normal(self.contact_point))):
            raise ValueError(f"{name} should be perpendicular to the normal vector.")
        self._longitudinal_axis = axis

    @property
    def lateral_axis(self) -> Vector:
        """Lateral axis of the wheel."""
        if self._lateral_axis is None:
            self._lateral_axis = cross(
                cross(self.ground.get_normal(self.contact_point),
                      self.wheel.rotation_axis),
                self.ground.get_normal(self.contact_point)).normalize()
        return self._lateral_axis

    @lateral_axis.setter
    def lateral_axis(self, axis: Vector) -> None:
        name = "The lateral axis of the wheel"
        if not isinstance(axis, Vector):
            raise TypeError(f"{name} should be a vector, but received a {type(axis)}")
        if not check_zero(axis.magnitude() - 1):
            raise ValueError(f"{name} should be normalized.")
        if not check_zero(axis.dot(self.longitudinal_axis)):
            raise ValueError(
                f"{name} should be perpendicular to the longitudinal axis.")
        if not check_zero(axis.dot(self.ground.get_normal(self.contact_point))):
            raise ValueError(f"{name} should be perpendicular to the normal vector.")
        self._lateral_axis = axis

    @property
    def contact_point(self) -> Point:
        """Point representing the contact point of the wheel with the ground."""
        return self._contact_point

    @property
    def on_ground(self) -> bool:
        """Boolean whether the wheel is already defined as touching the ground."""
        if self._on_ground is None:
            try:
                normal = self.ground.get_normal(self.contact_point)
                distance = self.contact_point.pos_from(self.ground.origin).dot(normal)
                self._on_ground = check_zero(distance)
            except (AttributeError, ValueError):
                self._on_ground = False
        return self._on_ground

    @on_ground.setter
    def on_ground(self, value: bool) -> None:
        self._on_ground = bool(value)


class InContactTire(TireBase):
    """Generic tire model that is in contact with the ground.

    Explanation
    -----------
    This tire model is a generic tire model that is defined to be in contact with the
    ground. It is used as a base class for other tire models that are in contact with
    the ground.

    Attributes
    ----------
    compute_normal_force : bool
        Flag to indicate if the normal force of the tire model should be computed.
        Default is True.
    no_longitudinal_slip : bool
        Flag to indicate if the tire model has no longitudinal slip. If True, a
        nonholonomic constraint is added to the system to enforce no slip in the
        longitudinal direction. Default is False.
    no_lateral_slip : bool
        Flag to indicate if the tire model has no lateral slip. If True, a nonholonomic
        constraint is added to the system to enforce no slip in the lateral direction.
        Default is False.
    """

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("ground", FlatGround, "Submodel of the ground."),
        ModelRequirement("wheel", (KnifeEdgeWheel, ToroidalWheel),
                         "Submodel of the wheel."),
    )
    ground: FlatGround
    wheel: KnifeEdgeWheel | ToroidalWheel

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.compute_normal_force: bool = True
        self.no_longitudinal_slip: bool = False
        self.no_lateral_slip: bool = False

    @property
    def descriptions(self) -> dict[Any, str]:
        """Dictionary of descriptions of the lateral tire model's attributes."""
        descriptions = super().descriptions
        if self.compute_normal_force:
            descriptions[self.u_aux[0]] = (
                f"Auxiliary generalized speed to determine the normal force of "
                f"'{self.name}'.")
        load_descriptions = {
            "Fx": f"Longitudinal force of the tire model '{self.name}'.",
            "Fy": f"Lateral force of the tire model '{self.name}'.",
            "Fz": f"Normal force of the tire model '{self.name}'.",
            "Mx": f"Rolling resistance moment of the tire model '{self.name}'.",
            "Mz": f"Self aligning moment of the tire model '{self.name}'.",
        }
        for name, description in load_descriptions.items():
            if name in self.symbols:
                descriptions[self.symbols[name]] = description
        return descriptions

    @property
    def camber_angle(self) -> Expr:
        """Camber angle of the wheel."""
        # atan2 is used instead of acos to account for the sign of the angle.
        # atan2 is used for consistency with the slip angle.
        return atan2(
            self.upward_radial_axis.dot(self.lateral_axis),
            self.upward_radial_axis.dot(self.ground.get_normal(self.contact_point)))

    @property
    def slip_angle(self) -> Expr:
        """Slip angle of the wheel."""
        # atan2 is used instead of acos to account for the sign of the angle.
        # atan2 is used instead of atan as the slip angle can be between -pi and pi.
        vel = self.contact_point.vel(self.ground.frame)
        return atan2(vel.dot(-self.lateral_axis), vel.dot(self.longitudinal_axis))

    def _define_objects(self) -> None:
        """Define the objects of the tire model."""
        super()._define_objects()
        syms_to_add = ["Fx", "Fy", "Mx", "Mz"]
        if self.compute_normal_force:
            self.u_aux = Matrix([dynamicsymbols(self._add_prefix("uaux_z"))])
            syms_to_add.append("Fz")
        if self.no_longitudinal_slip:
            syms_to_add.remove("Fx")
        if self.no_lateral_slip:
            syms_to_add.remove("Fy")
            syms_to_add.remove("Mz")
        if self.no_lateral_slip and self.no_longitudinal_slip:
            syms_to_add.remove("Mx")
        self.symbols.update({
            name: dynamicsymbols(self._add_prefix(name)) for name in syms_to_add
        })

    def _define_kinematics(self) -> None:
        """Define the kinematics of the tire model."""
        super()._define_kinematics()
        self._set_pos_contact_point()
        if (self.no_longitudinal_slip and self.no_lateral_slip
            and self.ground.origin in self.contact_point._pos_dict):
            self.contact_point.set_vel(self.ground.frame, 0)
            self.wheel.center.set_vel(
                self.ground.frame,
                -cross(self.wheel.center.pos_from(self.contact_point),
                       self.wheel.frame.ang_vel_in(self.ground.frame)))

        if self.compute_normal_force:
            direction = -self.ground.get_normal(self.contact_point)
            if self.on_ground:
                direction = -direction
            self.auxiliary_handler.add_noncontributing_force(
                self.contact_point, direction, self.u_aux[0], self.symbols["Fz"])

    def _define_loads(self) -> None:
        """Define the loads of the tire model."""
        super()._define_loads()
        tire_force = (
            self.symbols.get("Fx", 0) * self.longitudinal_axis +
            self.symbols.get("Fy", 0) * self.lateral_axis
        )
        if tire_force != 0:
            self.system.add_loads(Force(self.contact_point, tire_force))
        tire_torque = (
            self.symbols.get("Mx", 0) * self.longitudinal_axis +
            self.symbols.get("Mz", 0) * -self.ground.get_normal(self.contact_point)
        )
        if tire_torque != 0:
            self.system.add_loads(Torque(self.wheel.frame, tire_torque))

    def _define_constraints(self) -> None:
        """Define the constraints of the tire model."""
        super()._define_constraints()
        if self.no_longitudinal_slip and self.no_lateral_slip:
            no_slip_vectors = self.ground.get_tangent_vectors(self.contact_point)
        elif self.no_longitudinal_slip:
            no_slip_vectors = (self.longitudinal_axis,)
        elif self.no_lateral_slip:
            no_slip_vectors = (self.lateral_axis,)
        aux_cp = self.auxiliary_handler.get_auxiliary_velocity(self.contact_point)
        aux_gnd = self.auxiliary_handler.get_auxiliary_velocity(self.ground.origin)
        if self.no_longitudinal_slip or self.no_lateral_slip:
            aux_wc = self.auxiliary_handler.get_auxiliary_velocity(self.wheel.center)
            v0 = (self.wheel.center.pos_from(self.ground.origin).dt(self.ground.frame)
                  - cross(self.contact_point.pos_from(self.wheel.center),
                          self.wheel.frame.ang_vel_in(self.ground.frame)))
            aux_v0 = aux_wc - aux_gnd + aux_cp
            for vector in no_slip_vectors:
                self.system.add_nonholonomic_constraints(
                    v0.dot(vector) + aux_v0.dot(vector))
        if not self.on_ground:
            normal = self.ground.get_normal(self.contact_point)
            self.system.add_holonomic_constraints(
                self.contact_point.pos_from(self.ground.origin).dot(normal))
            self.system.velocity_constraints = [
                self.system.holonomic_constraints[0].diff(dynamicsymbols._t) +
                (aux_cp - aux_gnd).dot(normal), *self.system.nonholonomic_constraints
            ]


class NonHolonomicTire(InContactTire):
    """Tire model connection based on nonholonomic constraints."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.compute_normal_force = False
        self.no_lateral_slip = True
        self.no_longitudinal_slip = True
