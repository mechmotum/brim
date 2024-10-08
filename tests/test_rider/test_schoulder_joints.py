from __future__ import annotations

import pytest

from symbrim.rider.arms import PinElbowStickLeftArm, PinElbowStickRightArm
from symbrim.rider.base_connections import (
    LeftShoulderBase,
    RightShoulderBase,
)
from symbrim.rider.shoulder_joints import (
    FlexAddLeftShoulder,
    FlexAddRightShoulder,
    FlexRotLeftShoulder,
    FlexRotRightShoulder,
    SphericalLeftShoulder,
    SphericalRightShoulder,
    SphericalShoulderSpringDamper,
    SphericalShoulderTorque,
)
from symbrim.rider.torso import PlanarTorso
from symbrim.utilities.testing import _test_descriptions, create_model_of_connection
from symbrim.utilities.utilities import check_zero


@pytest.mark.parametrize("shoulder_cls, arm_cls, base_cls", [
    (FlexAddLeftShoulder, PinElbowStickLeftArm, LeftShoulderBase),
    (FlexAddRightShoulder, PinElbowStickRightArm, RightShoulderBase),
    (FlexRotLeftShoulder, PinElbowStickLeftArm, LeftShoulderBase),
    (FlexRotRightShoulder, PinElbowStickRightArm, RightShoulderBase),
    (SphericalLeftShoulder, PinElbowStickLeftArm, LeftShoulderBase),
    (SphericalRightShoulder, PinElbowStickRightArm, RightShoulderBase),
])
class TestShoulderJointBase:
    @pytest.fixture(autouse=True)
    def _setup(self, shoulder_cls, arm_cls, base_cls) -> None:
        self.shoulder = shoulder_cls("shoulder")
        self.shoulder.torso = PlanarTorso("torso")
        self.shoulder.arm = arm_cls("arm")

    def test_types(self, base_cls) -> None:
        assert isinstance(self.shoulder, base_cls)

    def test_descriptions(self) -> None:
        _test_descriptions(self.shoulder)


class ShoulderSetupMixin:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(self.shoulder_cls)("model")
        self.model.conn = self.shoulder_cls("shoulder")
        self.model.torso = PlanarTorso("torso")
        if issubclass(self.shoulder_cls, LeftShoulderBase):
            self.model.arm = PinElbowStickLeftArm("arm")
        else:
            self.model.arm = PinElbowStickRightArm("arm")
        self.model.define_all()
        self.shoulder, self.torso, self.arm = (
            self.model.conn, self.model.torso, self.model.arm)


class TestFlexAddLeftShoulder(ShoulderSetupMixin):
    shoulder_cls = FlexAddLeftShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.x).xreplace(
            {self.shoulder.q[0]: 0}) == -self.shoulder.u[1]


class TestFlexAddRightShoulder(ShoulderSetupMixin):
    shoulder_cls = FlexAddRightShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.x).xreplace(
            {self.shoulder.q[0]: 0}) == self.shoulder.u[1]


class TestFlexRotLeftShoulder(ShoulderSetupMixin):
    shoulder_cls = FlexRotLeftShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.z).xreplace(
            {self.shoulder.q[0]: 0}) == self.shoulder.u[1]


class TestFlexRotRightShoulder(ShoulderSetupMixin):
    shoulder_cls = FlexRotRightShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.z).xreplace(
            {self.shoulder.q[0]: 0}) == -self.shoulder.u[1]


class TestSphericalLeftShoulder(ShoulderSetupMixin):
    shoulder_cls = SphericalLeftShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0, self.shoulder.q[2]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.x).xreplace(
            {self.shoulder.q[0]: 0, self.shoulder.q[2]: 0}) == -self.shoulder.u[1]
        assert w.dot(self.torso.z).xreplace(
            {self.shoulder.q[0]: 0, self.shoulder.q[1]: 0}) == self.shoulder.u[2]


class TestSphericalRightShoulder(ShoulderSetupMixin):
    shoulder_cls = SphericalRightShoulder

    def test_kinematics(self) -> None:
        w = self.arm.upper_arm.frame.ang_vel_in(self.torso.frame)
        assert w.dot(self.torso.y).xreplace(
            {self.shoulder.q[1]: 0, self.shoulder.q[2]: 0}) == self.shoulder.u[0]
        assert w.dot(self.torso.x).xreplace(
            {self.shoulder.q[0]: 0, self.shoulder.q[2]: 0}) == self.shoulder.u[1]
        assert w.dot(self.torso.z).xreplace(
            {self.shoulder.q[0]: 0, self.shoulder.q[1]: 0}) == -self.shoulder.u[2]


class TestSphericalShoulderTorques:
    @pytest.mark.parametrize("load_group_cls", [
        SphericalShoulderTorque, SphericalShoulderSpringDamper])
    def test_invalid_type(self, load_group_cls) -> None:
        with pytest.raises(TypeError):
            PlanarTorso("torso").add_load_groups(load_group_cls("shoulder"))

    @pytest.mark.parametrize("load_group_cls", [
        SphericalShoulderTorque, SphericalShoulderSpringDamper])
    def test_descriptions(self, load_group_cls) -> None:
        _test_descriptions(load_group_cls("shoulder"))

    @staticmethod
    def _get_test_torques_info(shoulder_cls: type, arm_cls: type, load_group_cls: type
                               ) -> tuple:
        model = create_model_of_connection(shoulder_cls)("model")
        model.conn = shoulder_cls("shoulder")
        model.torso = PlanarTorso("torso")
        model.arm = arm_cls("arm")
        load_group = load_group_cls("shoulder")
        model.conn.add_load_groups(load_group)
        model.define_all()
        assert len(load_group.system.loads) == 2
        w = model.arm.shoulder_interframe.ang_vel_in(model.torso.frame)
        flex_axis = w.xreplace({model.conn.u[1]: 0, model.conn.u[2]: 0}).normalize()
        add_axis = w.xreplace({model.conn.u[0]: 0, model.conn.u[2]: 0}).normalize()
        rot_axis = w.xreplace({model.conn.u[0]: 0, model.conn.u[1]: 0}).normalize()
        return model, load_group, flex_axis, add_axis, rot_axis

    @pytest.mark.parametrize(("shoulder_cls", "leg_cls"), [
        (SphericalLeftShoulder, PinElbowStickLeftArm),
        (SphericalRightShoulder, PinElbowStickRightArm)])
    def test_torque_loads(self, shoulder_cls, leg_cls) -> None:
        model, load_group, flex_axis, add_axis, rot_axis = self._get_test_torques_info(
            shoulder_cls, leg_cls, SphericalShoulderTorque)
        t_flex, t_add, t_rot = (load_group.symbols[name] for name in (
            "T_flexion", "T_adduction", "T_rotation"))
        for load in load_group.system.loads:
            if load.frame == model.arm.shoulder_interframe:
                assert check_zero(
                    load.torque.xreplace({t_add: 0, t_rot: 0}).dot(flex_axis) - t_flex)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_rot: 0}).dot(add_axis) - t_add)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_add: 0}).dot(rot_axis) - t_rot)
            else:
                assert load.frame == model.torso.frame
                assert check_zero(
                    load.torque.xreplace({t_add: 0, t_rot: 0}).dot(flex_axis) - -t_flex)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_rot: 0}).dot(add_axis) - -t_add)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_add: 0}).dot(rot_axis) - -t_rot)

    @pytest.mark.parametrize(("shoulder_cls", "leg_cls"), [
        (SphericalLeftShoulder, PinElbowStickLeftArm),
        (SphericalRightShoulder, PinElbowStickRightArm)])
    def test_spring_damper_loads(self, shoulder_cls, leg_cls) -> None:
        model, load_group, flex_axis, add_axis, rot_axis = self._get_test_torques_info(
            shoulder_cls, leg_cls, SphericalShoulderSpringDamper)
        syms = [tuple(load_group.symbols[f"{tp}_{name}"] for tp in ("k", "c", "q_ref"))
                for name in ("flexion", "adduction", "rotation")]
        zero = [{sym: 0 for sym in syms_tp} for syms_tp in syms]

        def torque(syms, q, u):
            return -syms[0] * (q - syms[2]) - syms[1] * u

        for load in load_group.system.loads:
            if load.frame == model.arm.shoulder_interframe:
                assert check_zero(
                    load.torque.xreplace({**zero[1], **zero[2]}).dot(flex_axis) -
                    torque(syms[0], model.conn.q[0], model.conn.u[0]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[2]}).dot(add_axis) -
                    torque(syms[1], model.conn.q[1], model.conn.u[1]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[1]}).dot(rot_axis) -
                    torque(syms[2], model.conn.q[2], model.conn.u[2]))
            else:
                assert load.frame == model.torso.frame
                assert check_zero(
                    load.torque.xreplace({**zero[1], **zero[2]}).dot(flex_axis) +
                    torque(syms[0], model.conn.q[0], model.conn.u[0]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[2]}).dot(add_axis) +
                    torque(syms[1], model.conn.q[1], model.conn.u[1]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[1]}).dot(rot_axis) +
                    torque(syms[2], model.conn.q[2], model.conn.u[2]))
