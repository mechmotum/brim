from __future__ import annotations

import pytest
from sympy import Symbol
from sympy.physics.mechanics import Vector, dynamicsymbols

from symbrim.bicycle.cranks import MasslessCranks
from symbrim.brim.base_connections import PedalsBase
from symbrim.brim.pedals import HolonomicPedals, SpringDamperPedals
from symbrim.rider.legs import TwoPinStickLeftLeg, TwoPinStickRightLeg
from symbrim.utilities.testing import _test_descriptions, create_model_of_connection


@pytest.mark.parametrize("pedal_cls", [HolonomicPedals, SpringDamperPedals])
class TestPedalsBase:
    @pytest.fixture
    def _setup(self, pedal_cls) -> None:
        self.model = create_model_of_connection(pedal_cls)("model")
        self.model.cranks = MasslessCranks("cranks")
        self.model.left_leg = TwoPinStickLeftLeg("left_leg")
        self.model.right_leg = TwoPinStickRightLeg("right_leg")
        self.model.conn = pedal_cls("pedal_connection")
        self.model.define_connections()
        self.model.define_objects()
        # Define kinematics with enough degrees of freedom
        self.q = dynamicsymbols("q1:5")
        self.model.left_leg.hip_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, self.q[0])
        self.model.left_leg.hip_interpoint.set_pos(
            self.model.cranks.left_pedal_point,
            self.q[1] * self.model.cranks.rotation_axis)
        self.model.right_leg.hip_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, self.q[2])
        self.model.right_leg.hip_interpoint.set_pos(
            self.model.cranks.right_pedal_point,
            self.q[3] * self.model.cranks.rotation_axis)
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()

    @pytest.mark.usefixtures("_setup")
    def test_types(self) -> None:
        assert isinstance(self.model.conn, PedalsBase)

    @pytest.mark.usefixtures("_setup")
    def test_descriptions(self) -> None:
        _test_descriptions(self.model.conn)

    @pytest.mark.parametrize(("side", "leg_cls"), [
        ("left", TwoPinStickLeftLeg), ("right", TwoPinStickRightLeg)])
    def test_single_leg(self, pedal_cls, side, leg_cls) -> None:
        self.model = create_model_of_connection(pedal_cls)("model")
        self.model.cranks = MasslessCranks("cranks")
        self.model.conn = pedal_cls("pedals")
        setattr(self.model, f"{side}_leg", leg_cls(f"{side}_leg"))
        self.model.define_connections()
        self.model.define_objects()
        # Define kinematics with enough degrees of freedom
        self.q = dynamicsymbols("q1:3")
        getattr(self.model, f"{side}_leg").hip_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, self.q[0])
        getattr(self.model, f"{side}_leg").hip_interpoint.set_pos(
            self.model.cranks.left_pedal_point,
            self.q[1] * self.model.cranks.rotation_axis)
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()


class TestHolonomicPedals:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(HolonomicPedals)("model")
        self.model.cranks = MasslessCranks("cranks")
        self.model.left_leg = TwoPinStickLeftLeg("left_leg")
        self.model.right_leg = TwoPinStickRightLeg("right_leg")
        self.model.conn = HolonomicPedals("pedal_connection")
        self.model.define_connections()
        self.model.define_objects()
        self.model.left_leg.foot_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, 0)
        self.model.right_leg.foot_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, 0)
        self.cranks, self.left_leg, self.right_leg, self.conn = (
            self.model.cranks, self.model.left_leg, self.model.right_leg,
            self.model.conn)

    def test_all_constraints(self) -> None:
        q = dynamicsymbols("q1:7")
        self.left_leg.foot_interpoint.set_pos(
            self.cranks.left_pedal_point,
            sum(qi * v for qi, v in zip(q[:3], self.cranks.frame)))
        self.right_leg.foot_interpoint.set_pos(
            self.cranks.right_pedal_point,
            sum(qi * v for qi, v in zip(q[3:], self.cranks.frame)))
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()
        assert len(self.conn.system.holonomic_constraints) == 6
        expected_constraints = list(q)
        for constr in self.conn.system.holonomic_constraints:
            for exp_constr in expected_constraints:
                if constr.xreplace({exp_constr: 0}) == 0:
                    expected_constraints.remove(exp_constr)
        assert not expected_constraints

    def test_not_fully_constraint(self) -> None:
        q, d = dynamicsymbols("q"), Symbol("d")
        self.left_leg.foot_interpoint.set_pos(self.cranks.left_pedal_point, 0)
        self.right_leg.foot_interpoint.set_pos(
            self.cranks.right_pedal_point,
            q * self.cranks.frame.x + 2 * d * self.cranks.frame.x)
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()
        assert len(self.conn.system.holonomic_constraints) == 1
        assert self.conn.system.holonomic_constraints[0].xreplace({q: 0.6, d: -0.3}
                                                                  ) == 0

    def test_constant_constraint(self) -> None:
        q, d = dynamicsymbols("q"), Symbol("d")
        self.left_leg.foot_interpoint.set_pos(self.cranks.left_pedal_point, 0)
        self.right_leg.foot_interpoint.set_pos(
            self.cranks.right_pedal_point,
            q * self.cranks.frame.x + d * self.cranks.frame.y)
        self.model.define_kinematics()
        self.model.define_loads()
        with pytest.raises(ValueError):
            self.model.define_constraints()


class TestSpringDamperPedals:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(SpringDamperPedals)("model")
        self.model.cranks = MasslessCranks("cranks")
        self.model.left_leg = TwoPinStickLeftLeg("left_leg")
        self.model.right_leg = TwoPinStickRightLeg("right_leg")
        self.model.conn = SpringDamperPedals("pedal_connection")
        self.model.define_connections()
        self.model.define_objects()
        self.model.left_leg.foot_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, 0)
        self.model.right_leg.foot_interframe.orient_axis(
            self.model.cranks.frame, self.model.cranks.rotation_axis, 0)
        self.cranks, self.left_leg, self.right_leg, self.conn = (
            self.model.cranks, self.model.left_leg, self.model.right_leg,
            self.model.conn)

    def test_loads(self) -> None:
        q1, q2 = dynamicsymbols("q1:3")
        self.left_leg.foot_interpoint.set_pos(
            self.cranks.left_pedal_point, q1 * self.cranks.frame.x)
        self.right_leg.foot_interpoint.set_pos(
            self.cranks.right_pedal_point, -q2 * self.cranks.frame.y)
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()
        loads_indi = [ld for act in self.conn.system.actuators for ld in act.to_loads()]
        locations, loads = {}, []
        for ld in loads_indi:
            if ld.location not in locations:
                locations[ld.location] = len(loads)
                loads.append(ld)
            else:
                loads[locations[ld.location]] = ld.__class__(
                    ld.location, ld.vector + loads[locations[ld.location]].vector)
        assert len(loads) == 4
        k, c = self.conn.symbols["k"], self.conn.symbols["c"]
        for ld in loads:
            if ld.location == self.cranks.left_pedal_point:
                assert (ld.vector - (k * q1 + c * q1.diff()) * self.cranks.frame.x
                        ).simplify() == Vector(0)
            elif ld.location == self.left_leg.foot_interpoint:
                assert (ld.vector + (k * q1 + c * q1.diff()) * self.cranks.frame.x
                        ).simplify() == Vector(0)
            elif ld.location == self.cranks.right_pedal_point:
                assert (ld.vector - (-k * q2 - c * q2.diff()) * self.cranks.frame.y
                        ).simplify() == Vector(0)
            else:
                assert ld.location == self.right_leg.foot_interpoint
                assert (ld.vector + (-k * q2 - c * q2.diff()) * self.cranks.frame.y
                        ).simplify() == Vector(0)
