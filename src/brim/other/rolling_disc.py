"""Module containing the rolling disc model."""
from __future__ import annotations

import contextlib
from typing import Any

from sympy import Matrix, Symbol, symbols
from sympy.physics.mechanics import (
    ReferenceFrame,
    RigidBody,
    System,
    cross,
    dynamicsymbols,
    inertia,
)

from brim.bicycle.grounds import GroundBase
from brim.bicycle.tires import TireBase
from brim.bicycle.wheels import WheelBase
from brim.core import ConnectionRequirement, ModelBase, ModelRequirement


class RollingDisc(ModelBase):
    """Rolling disc model."""

    required_models: tuple[ModelRequirement, ...] = (
        ModelRequirement("ground", GroundBase, "Ground model."),
        ModelRequirement("disc", WheelBase, "Disc model."),
    )
    required_connections: tuple[ConnectionRequirement, ...] = (
        ConnectionRequirement("tire", TireBase, "Tire model."),
    )
    ground: GroundBase
    disc: WheelBase
    tire: TireBase

    @property
    def descriptions(self) -> dict[Any, str]:
        """Dictionary of descriptions of the rolling disc's attributes."""
        desc = {
            **super().descriptions,
            self.q[0]: "Perpendicular distance along ground.x to the contact point in "
                       "the ground plane.",
            self.q[1]: "Perpendicular distance along ground.y to the contact point in "
                       "the ground plane.",
            self.q[2]: "Yaw angle of the disc.",
            self.q[3]: "Roll angle of the disc.",
            self.q[4]: "Pitch angle of the disc.",
        }
        desc.update({ui: f"Generalized speed of the {desc[qi].lower()}"
                     for qi, ui in zip(self.q, self.u)})
        return desc

    def _define_connections(self) -> None:
        """Define the connections between the submodels."""
        super()._define_connections()
        self.tire.ground = self.ground
        self.tire.wheel = self.disc

    def _define_objects(self) -> None:
        """Define the objects of the rolling disc."""
        super()._define_objects()
        self._system = System(self.ground.frame, self.ground.origin)
        self.tire.define_objects()
        self.tire.on_ground = True
        self.q = Matrix([dynamicsymbols(self._add_prefix("q1:6"))])
        self.u = Matrix([dynamicsymbols(self._add_prefix("u1:6"))])

    def _define_kinematics(self) -> None:
        """Define the kinematics of the rolling disc."""
        super()._define_kinematics()
        qd_repl = dict(zip(self.q.diff(dynamicsymbols._t), self.u))
        yaw_frame = ReferenceFrame("yaw_frame")
        roll_frame = ReferenceFrame("roll_frame")
        yaw_frame.orient_axis(self.ground.frame, self.ground.frame.z, self.q[2])
        roll_frame.orient_axis(yaw_frame, yaw_frame.x, self.q[3])
        self.disc.frame.orient_axis(roll_frame, roll_frame.y, self.q[4])
        self.disc.frame.set_ang_vel(
            self.ground.frame, self.disc.frame.ang_vel_in(self.ground.frame).xreplace(
                qd_repl))
        self.ground.set_pos_point(self.tire.contact_point, self.q[:2])
        self.tire.contact_point.set_vel(
            self.ground.frame,
            self.tire.contact_point.vel(self.ground.frame).xreplace(qd_repl))
        with contextlib.suppress(ValueError):
            normal = self.ground.get_normal(self.tire.contact_point)
            direction = normal.dot(-self.ground.frame.z)
            self.tire.upward_radial_axis = direction * -roll_frame.z
            self.tire.longitudinal_axis = direction * yaw_frame.x
            self.tire.lateral_axis = yaw_frame.y

        self.tire.define_kinematics()
        self.system.q_ind = self.q
        self.system.u_ind = self.u
        self.system.kdes = [qdi - ui for qdi, ui in qd_repl.items()]

    def _define_loads(self) -> None:
        """Define the loads of the rolling disc."""
        super()._define_loads()
        self.tire.define_loads()

    def _define_constraints(self) -> None:
        """Define the constraints of the rolling disc."""
        super()._define_constraints()
        self.tire.define_constraints()


def rolling_disc_manual() -> System:
    """Create a rolling disc model manually.

    Notes
    -----
    This function is used to verify and benchmark the rolling disc model. It is mostly
    copied from _create_rolling_disc in test_kane5.py from SymPy.
    """
    # Define symbols and coordinates
    t = dynamicsymbols._t
    q = dynamicsymbols("q1:6")
    u = dynamicsymbols("u1:6")
    qd_repl = {qi.diff(t): ui for qi, ui in zip(q, u)}
    # Define bodies and frames
    ground = RigidBody("ground")
    disc = RigidBody("disc", mass=Symbol("m"))
    disc.inertia = (inertia(disc.frame, *symbols("ixx iyy ixx")), disc.masscenter)
    ground.masscenter.set_vel(ground.frame, 0)
    disc.masscenter.set_vel(disc.frame, 0)
    int_frame = ReferenceFrame("int_frame")

    # Orient frames
    int_frame.orient_body_fixed(ground.frame, (q[2], q[3], 0), "zxy")
    disc.frame.orient_axis(int_frame, int_frame.y, q[4])
    g_w_d = disc.frame.ang_vel_in(ground.frame)
    disc.frame.set_ang_vel(ground.frame, g_w_d.xreplace(qd_repl))
    # Define points
    cp = ground.masscenter.locatenew("contact_point",
                                     q[0] * ground.x + q[1] * ground.y)
    cp.set_vel(ground.frame, u[0] * ground.x + u[1] * ground.y)
    disc.masscenter.set_pos(cp, -Symbol("r") * int_frame.z)

    # Define kinematic differential equations
    kdes = [qdi - ui for qdi, ui in qd_repl.items()]
    # Define nonholonomic constraints
    v0 = disc.masscenter.vel(ground.frame) + cross(
        disc.frame.ang_vel_in(ground.frame), cp.pos_from(disc.masscenter))
    fnh = [v0.dot(ground.x), v0.dot(ground.y)]
    # Create system
    system = System.from_newtonian(ground)
    system.q_ind = q
    system.u_ind = u[2:]
    system.u_dep = u[:2]
    system.kdes = kdes
    system.nonholonomic_constraints = fnh
    system.bodies = disc
    system.loads = [(disc.masscenter, disc.mass * Symbol("g") * ground.z)]
    return system
