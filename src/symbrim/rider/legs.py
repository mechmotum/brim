"""Module containing models of the legs."""
from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import TYPE_CHECKING

from sympy import Matrix, Symbol
from sympy.physics.mechanics import (
    PinJoint,
    Point,
    ReferenceFrame,
    RigidBody,
    System,
    TorqueActuator,
    dynamicsymbols,
)

from symbrim.core import LoadGroupBase, ModelBase

with contextlib.suppress(ImportError):
    import numpy as np
    from yeadon.inertia import rotate_inertia

    from symbrim.utilities.parametrize import get_inertia_vals_from_yeadon

    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel

__all__ = ["LegBase", "LeftLegBase", "RightLegBase", "TwoPinStickLeftLeg",
           "TwoPinStickRightLeg", "TwoPinLegTorque", "TwoPinLegSpringDamper"]


class LegBase(ModelBase):
    """Base class for the leg of the rider."""

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self._hip_interpoint = Point(self._add_prefix("HP"))
        self._foot_interpoint = Point(self._add_prefix("FP"))

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        self.hip_interpoint.set_vel(self.hip_interframe, 0)
        self.foot_interpoint.set_vel(self.foot_interframe, 0)

    @property
    @abstractmethod
    def hip(self) -> RigidBody:
        """Hip of the leg."""

    @property
    def hip_interpoint(self) -> Point:
        """Point where the hip is attached to the pelvis."""
        return self._hip_interpoint

    @property
    @abstractmethod
    def hip_interframe(self) -> ReferenceFrame:
        """Frame to be used as inteframe between the hip and the pelvis."""

    @property
    @abstractmethod
    def foot(self) -> RigidBody:
        """Foot of the leg."""

    @property
    def foot_interpoint(self) -> Point:
        """Point where the foot is attached to the pedals."""
        return self._foot_interpoint

    @property
    @abstractmethod
    def foot_interframe(self) -> ReferenceFrame:
        """Frame to be used as inteframe between the foot and the pedals."""

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_line([
            self.hip_interpoint,
            *(joint.parent_point for joint in self.system.joints),
            self.foot_interpoint],
            self.name)


class LeftLegBase(LegBase):
    """Base class for the left leg of the rider."""


class RightLegBase(LegBase):
    """Base class for the right leg of the rider."""


class TwoPinStickLegMixin:
    """Mixin class for a leg with two pin joints."""

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["l_thigh"]: "Length of the thigh.",
            self.symbols["l_thigh_com"]: "Distance from the hip joint to the center of "
                                         "mass of the thigh.",
            self.symbols["l_shank"]: "Length of the shank.",
            self.symbols["l_shank_com"]: "Distance from the knee joint to the center of"
                                         " mass of the shank.",
            self.symbols["l_foot"]: "Length of the foot.",
            self.symbols["l_foot_com"]: "Distance from the ankle joint to the center of"
                                        " mass of the foot.",
            self.q[0]: "Knee flexion angle.",
            self.q[1]: "Ankle flexion angle.",
            self.u[0]: "Knee flexion rate.",
            self.u[1]: "Ankle flexion rate.",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        super()._define_objects()
        self.symbols.update({
            "l_thigh": Symbol(self._add_prefix("l_thigh")),
            "l_thigh_com": Symbol(self._add_prefix("l_thigh_com")),
            "l_shank": Symbol(self._add_prefix("l_shank")),
            "l_shank_com": Symbol(self._add_prefix("l_shank_com")),
            "l_foot": Symbol(self._add_prefix("l_foot")),
            "l_foot_com": Symbol(self._add_prefix("l_foot_com")),
        })
        self.q = Matrix(
            dynamicsymbols(self._add_prefix("q_knee_flexion q_ankle_flexion")))
        self.u = Matrix(
            dynamicsymbols(self._add_prefix("u_knee_flexion u_ankle_flexion")))
        self._thigh = RigidBody(self._add_prefix("thigh"))
        self._shank = RigidBody(self._add_prefix("shank"))
        self._foot = RigidBody(self._add_prefix("foot"))
        self._system = System.from_newtonian(self.hip)
        self._system.add_bodies(self._shank, self._foot)

    def _define_kinematics(self) -> None:
        """Define the kinematics."""
        super()._define_kinematics()
        l_f, l_t, l_s, l_f_com, l_t_com, l_s_com = (self.symbols[length] for length in (
            "l_foot", "l_thigh", "l_shank", "l_foot_com", "l_thigh_com", "l_shank_com"))
        self.hip.masscenter.set_pos(self.hip_interpoint, l_t_com * self.thigh.z)
        self.foot.masscenter.set_pos(self.foot_interpoint,
                                     (l_f_com - l_f) * self.foot.x)
        self.system.add_joints(
            PinJoint(
                self._add_prefix("knee"), self.thigh, self.shank, self.q[0], self.u[0],
                (l_t - l_t_com) * self.thigh.z, -l_s_com * self.shank.z,
                joint_axis=-self.thigh.y),
            PinJoint(
                self._add_prefix("ankle"), self.shank, self.foot, self.q[1], self.u[1],
                (l_s - l_s_com) * self.shank.z, -l_f_com * self.foot.x,
                joint_axis=-self.shank.y),
        )

    @property
    def hip(self) -> RigidBody:
        """Hip of the leg."""
        return self._thigh

    @property
    def hip_interframe(self) -> ReferenceFrame:
        """Frame to be used as inteframe between the hip and the pelvis."""
        return self._thigh.frame

    @property
    def foot_interframe(self) -> ReferenceFrame:
        """Frame to be used as inteframe between the foot and the pedals."""
        return self._foot.frame

    @property
    def thigh(self) -> RigidBody:
        """Thigh of the leg."""
        return self._thigh

    @property
    def shank(self) -> RigidBody:
        """Shank of the leg."""
        return self._shank

    @property
    def foot(self) -> RigidBody:
        """Foot of the leg."""
        return self._foot


class TwoPinStickLeftLeg(TwoPinStickLegMixin, LeftLegBase):
    """Left leg of the rider with two pin joints."""

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.thigh.mass] = human.J1.mass
        shank_props = human.combine_inertia(("j3", "j4", "j5"))
        foot_props = human.combine_inertia(("j6", "j7", "j8"))
        params[self.shank.mass] = shank_props[0]
        params[self.foot.mass] = foot_props[0]
        params.update(get_inertia_vals_from_yeadon(self.thigh, human.J1.rel_inertia))
        params.update(get_inertia_vals_from_yeadon(
            self.shank, rotate_inertia(human.J2.rot_mat, shank_props[2])))
        params.update(get_inertia_vals_from_yeadon(
            self.foot, rotate_inertia(human.J2.rot_mat, foot_props[2])))
        params.update({
            self.symbols["l_thigh"]: human.meas["Lj3L"],
            self.symbols["l_shank"]:
                human.meas["Lj5L"] - human.meas["Lj3L"] + human.meas["Lj6L"],
            self.symbols["l_foot"]: human.meas["Lj9L"] - human.meas["Lj6L"],
            self.symbols["l_thigh_com"]: -human.J1.rel_center_of_mass[2, 0],
            self.symbols["l_shank_com"]: np.linalg.norm(shank_props[1] - human.J2.pos),
            self.symbols["l_foot_com"]: np.linalg.norm(
                foot_props[1] - human.J2.solids[2].pos),
        })
        return params


class TwoPinStickRightLeg(TwoPinStickLegMixin, RightLegBase):
    """Right leg of the rider with two pin joints."""

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get the parameter values of the pelvis."""
        params = super().get_param_values(bicycle_parameters)
        human = bicycle_parameters.human
        if human is None:
            return params
        params[self.thigh.mass] = human.K1.mass
        shank_props = human.combine_inertia(("k3", "k4", "k5"))
        foot_props = human.combine_inertia(("k6", "k7", "k8"))
        params[self.shank.mass] = shank_props[0]
        params[self.foot.mass] = foot_props[0]
        params.update(get_inertia_vals_from_yeadon(self.thigh, human.K1.rel_inertia))
        params.update(get_inertia_vals_from_yeadon(
            self.shank, rotate_inertia(human.K2.rot_mat, shank_props[2])))
        params.update(get_inertia_vals_from_yeadon(
            self.foot, rotate_inertia(human.K2.rot_mat, foot_props[2])))
        params.update({
            self.symbols["l_thigh"]: human.meas["Lk3L"],
            self.symbols["l_shank"]:
                human.meas["Lk5L"] - human.meas["Lk3L"] + human.meas["Lk6L"],
            self.symbols["l_foot"]: human.meas["Lk9L"] - human.meas["Lk6L"],
            self.symbols["l_thigh_com"]: -human.K1.rel_center_of_mass[2, 0],
            self.symbols["l_shank_com"]: np.linalg.norm(shank_props[1] - human.K2.pos),
            self.symbols["l_foot_com"]: np.linalg.norm(
                foot_props[1] - human.K2.solids[2].pos),
        })
        return params


class TwoPinLegTorque(LoadGroupBase):
    """Torque applied to the leg of the rider as time-varying quantity."""

    parent: TwoPinStickLeftLeg | TwoPinStickRightLeg
    required_parent_type = (TwoPinStickLeftLeg, TwoPinStickRightLeg)

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["T_knee"]: f"Knee torque of {self.parent}",
            self.symbols["T_ankle"]: f"Ankle torque of {self.parent}",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols.update({name: dynamicsymbols(self._add_prefix(name)) for name in (
            "T_knee", "T_ankle")})

    def _define_loads(self) -> None:
        """Define the kinematics."""
        self.system.add_actuators(
            TorqueActuator(self.symbols["T_knee"], self.parent.thigh.y,
                           self.parent.shank, self.parent.thigh),
            TorqueActuator(self.symbols["T_ankle"], self.parent.shank.y,
                           self.parent.foot, self.parent.shank)
        )


class TwoPinLegSpringDamper(LoadGroupBase):
    """Torque applied to the leg of the rider as linear spring-damper."""

    parent: TwoPinStickLeftLeg | TwoPinStickRightLeg
    required_parent_type = (TwoPinStickLeftLeg, TwoPinStickRightLeg)

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["k_knee"]: f"Knee stiffness of {self.parent}",
            self.symbols["c_knee"]: f"Knee damping of {self.parent}",
            self.symbols["q_ref_knee"]: f"Knee reference angle of {self.parent}",
            self.symbols["k_ankle"]: f"Ankle stiffness of {self.parent}",
            self.symbols["c_ankle"]: f"Ankle damping of {self.parent}",
            self.symbols["q_ref_ankle"]: f"Ankle reference angle of {self.parent}",
        }

    def _define_objects(self) -> None:
        """Define the objects."""
        self.symbols.update({name: dynamicsymbols(self._add_prefix(name)) for name in (
            "k_knee", "c_knee", "q_ref_knee", "k_ankle", "c_ankle", "q_ref_ankle")})

    def _define_loads(self) -> None:
        """Define the kinematics."""
        knee = self.parent.system.get_joint(self.parent._add_prefix("knee"))
        ankle = self.parent.system.get_joint(self.parent._add_prefix("ankle"))
        q_knee, u_knee = knee.coordinates[0], knee.speeds[0]
        q_ankle, u_ankle = ankle.coordinates[0], ankle.speeds[0]
        self.system.add_actuators(
            TorqueActuator(
                -self.symbols["k_knee"] * (q_knee - self.symbols["q_ref_knee"]) -
                self.symbols["c_knee"] * u_knee,
                self.parent.thigh.y, self.parent.shank, self.parent.thigh),
            TorqueActuator(
                -self.symbols["k_ankle"] * (q_ankle - self.symbols["q_ref_ankle"]) -
                self.symbols["c_ankle"] * u_ankle,
                self.parent.shank.y, self.parent.foot, self.parent.shank)
        )
