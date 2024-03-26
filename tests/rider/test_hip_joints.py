from __future__ import annotations

import pytest
from brim.rider.base_connections import LeftHipBase, RightHipBase
from brim.rider.hip_joints import (
    PinLeftHip,
    PinRightHip,
    SphericalHipSpringDamper,
    SphericalHipTorque,
    SphericalLeftHip,
    SphericalRightHip,
)
from brim.rider.legs import TwoPinStickLeftLeg, TwoPinStickRightLeg
from brim.rider.pelvis import PlanarPelvis
from brim.utilities.testing import _test_descriptions, create_model_of_connection
from brim.utilities.utilities import check_zero


@pytest.mark.parametrize("hip_cls, leg_cls, base_cls", [
    (SphericalLeftHip, TwoPinStickLeftLeg, LeftHipBase),
    (SphericalRightHip, TwoPinStickRightLeg, RightHipBase),
    (PinLeftHip, TwoPinStickLeftLeg, LeftHipBase),
    (PinRightHip, TwoPinStickRightLeg, RightHipBase),
])
class TestHipJointBase:
    def test_types(self, hip_cls, leg_cls, base_cls) -> None:
        hip = hip_cls("hip")
        hip.pelvis = PlanarPelvis("pelvis")
        hip.leg = leg_cls("leg")
        assert isinstance(hip, base_cls)

    def test_descriptions(self, hip_cls, leg_cls, base_cls) -> None:
        hip = hip_cls("hip")
        hip.pelvis = PlanarPelvis("pelvis")
        hip.leg = leg_cls("leg")
        _test_descriptions(hip)


class TestSphericalLeftHipJoint:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(SphericalLeftHip)("hip_model")
        self.model.conn = SphericalLeftHip("hip")
        self.model.pelvis = PlanarPelvis("pelvis")
        self.model.leg = TwoPinStickLeftLeg("leg")
        self.model.define_all()
        self.hip, self.pelvis, self.leg = (
            self.model.conn, self.model.pelvis, self.model.leg)

    def test_kinematics(self):
        w = self.leg.hip_interframe.ang_vel_in(self.pelvis.frame)
        assert w.dot(self.pelvis.y).xreplace(
            {self.hip.q[1]: 0, self.hip.q[2]: 0}) == self.hip.u[0]
        assert w.dot(self.pelvis.x).xreplace(
            {self.hip.q[0]: 0, self.hip.q[2]: 0}) == -self.hip.u[1]
        assert w.dot(self.pelvis.z).xreplace(
            {self.hip.q[0]: 0, self.hip.q[1]: 0}) == self.hip.u[2]


class TestSphericalRightHipJoint:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(SphericalRightHip)("hip_model")
        self.model.conn = SphericalRightHip("hip")
        self.model.pelvis = PlanarPelvis("pelvis")
        self.model.leg = TwoPinStickRightLeg("leg")
        self.model.define_all()
        self.hip, self.pelvis, self.leg = (
            self.model.conn, self.model.pelvis, self.model.leg)

    def test_kinematics(self):
        w = self.leg.hip_interframe.ang_vel_in(self.pelvis.frame)
        assert w.dot(self.pelvis.y).xreplace(
            {self.hip.q[1]: 0, self.hip.q[2]: 0}) == self.hip.u[0]
        assert w.dot(self.pelvis.x).xreplace(
            {self.hip.q[0]: 0, self.hip.q[2]: 0}) == self.hip.u[1]
        assert w.dot(self.pelvis.z).xreplace(
            {self.hip.q[0]: 0, self.hip.q[1]: 0}) == -self.hip.u[2]


@pytest.mark.parametrize("hip_cls, leg_cls", [
    (PinLeftHip, TwoPinStickLeftLeg), (PinRightHip, TwoPinStickRightLeg)])
class TestPinLeftHipJoint:
    @pytest.fixture(autouse=True)
    def _setup(self, hip_cls, leg_cls) -> None:
        self.model = create_model_of_connection(hip_cls)("hip_model")
        self.model.conn = hip_cls("hip")
        self.model.pelvis = PlanarPelvis("pelvis")
        self.model.leg = leg_cls("leg")
        self.model.define_all()
        self.hip, self.pelvis, self.leg = (
            self.model.conn, self.model.pelvis, self.model.leg)

    def test_kinematics(self):
        w = self.leg.hip_interframe.ang_vel_in(self.pelvis.frame)
        assert w.dot(self.pelvis.y) == self.hip.u[0]


class TestSphericalHipTorque:
    @pytest.mark.parametrize("load_group_cls", [
        SphericalHipTorque, SphericalHipSpringDamper])
    def test_invalid_type(self, load_group_cls) -> None:
        with pytest.raises(TypeError):
            PinLeftHip("hip").add_load_groups(load_group_cls("hip"))

    @pytest.mark.parametrize("load_group_cls", [
        SphericalHipTorque, SphericalHipSpringDamper])
    def test_descriptions(self, load_group_cls) -> None:
        _test_descriptions(load_group_cls("hip"))

    @staticmethod
    def _get_test_torques_info(hip_cls: type, leg_cls: type, load_group_cls: type
                               ) -> tuple:
        model = create_model_of_connection(hip_cls)("hip_model")
        model.conn = hip_cls("hip")
        model.pelvis = PlanarPelvis("pelvis")
        model.leg = leg_cls("leg")
        load_group = load_group_cls("hip")
        model.conn.add_load_groups(load_group)
        model.define_all()
        assert len(load_group.system.loads) == 2
        w = model.leg.hip_interframe.ang_vel_in(model.pelvis.frame)
        flex_axis = w.xreplace({model.conn.u[1]: 0, model.conn.u[2]: 0}).normalize()
        add_axis = w.xreplace({model.conn.u[0]: 0, model.conn.u[2]: 0}).normalize()
        rot_axis = w.xreplace({model.conn.u[0]: 0, model.conn.u[1]: 0}).normalize()
        return model, load_group, flex_axis, add_axis, rot_axis

    @pytest.mark.parametrize("hip_cls, leg_cls", [
        (SphericalLeftHip, TwoPinStickLeftLeg),
        (SphericalRightHip, TwoPinStickRightLeg)])
    def test_torque_loads(self, hip_cls, leg_cls) -> None:
        model, load_group, flex_axis, add_axis, rot_axis = self._get_test_torques_info(
            hip_cls, leg_cls, SphericalHipTorque)
        t_flex, t_add, t_rot = (load_group.symbols[name] for name in (
            "T_flexion", "T_adduction", "T_rotation"))
        for load in load_group.system.loads:
            if load.frame == model.leg.thigh.frame:
                assert check_zero(
                    load.torque.xreplace({t_add: 0, t_rot: 0}).dot(flex_axis) - t_flex)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_rot: 0}).dot(add_axis) - t_add)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_add: 0}).dot(rot_axis) - t_rot)
            else:
                assert load.frame == model.pelvis.frame
                assert check_zero(
                    load.torque.xreplace({t_add: 0, t_rot: 0}).dot(flex_axis) - -t_flex)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_rot: 0}).dot(add_axis) - -t_add)
                assert check_zero(
                    load.torque.xreplace({t_flex: 0, t_add: 0}).dot(rot_axis) - -t_rot)

    @pytest.mark.parametrize("hip_cls, leg_cls", [
        (SphericalLeftHip, TwoPinStickLeftLeg),
        (SphericalRightHip, TwoPinStickRightLeg)])
    def test_spring_damper_loads(self, hip_cls, leg_cls) -> None:
        model, load_group, flex_axis, add_axis, rot_axis = self._get_test_torques_info(
            hip_cls, leg_cls, SphericalHipSpringDamper)
        syms = [tuple(load_group.symbols[f"{tp}_{name}"] for tp in ("k", "c", "q_ref"))
                for name in ("flexion", "adduction", "rotation")]
        zero = [{sym: 0 for sym in syms_tp} for syms_tp in syms]

        def torque(syms, q, u):
            return -syms[0] * (q - syms[2]) - syms[1] * u

        for load in load_group.system.loads:
            if load.frame == model.leg.thigh.frame:
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
                assert load.frame == model.pelvis.frame
                assert check_zero(
                    load.torque.xreplace({**zero[1], **zero[2]}).dot(flex_axis) +
                    torque(syms[0], model.conn.q[0], model.conn.u[0]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[2]}).dot(add_axis) +
                    torque(syms[1], model.conn.q[1], model.conn.u[1]))
                assert check_zero(
                    load.torque.xreplace({**zero[0], **zero[1]}).dot(rot_axis) +
                    torque(syms[2], model.conn.q[2], model.conn.u[2]))
