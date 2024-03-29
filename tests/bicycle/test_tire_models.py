from __future__ import annotations

import pytest
from brim.bicycle.grounds import FlatGround, GroundBase
from brim.bicycle.tires import NonHolonomicTire, TireBase
from brim.bicycle.wheels import KnifeEdgeWheel, ToroidalWheel, WheelBase
from brim.utilities.testing import create_model_of_connection
from sympy import cos, sin
from sympy.physics.mechanics import ReferenceFrame, System, cross, dynamicsymbols


class MyTire(TireBase):
    pass


class TestComputeContactPoint:
    @pytest.fixture()
    def _setup_flat_ground(self):
        self.ground = FlatGround("ground")
        self.ground.define_objects()
        self.ground.define_kinematics()
        self.q = dynamicsymbols("q1:4")
        self.int_frame = ReferenceFrame("int_frame")
        self.int_frame.orient_body_fixed(self.ground.frame, (*self.q[:2], 0), "zxy")
        self.tire = MyTire("tire")
        self.tire.ground = self.ground
        self.tire.define_objects()

    def test_knife_edge_wheel_on_flat_ground(self, _setup_flat_ground):
        wheel = KnifeEdgeWheel("wheel")
        wheel.define_objects()
        wheel.define_kinematics()
        self.tire.wheel = wheel
        wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        self.tire._set_pos_contact_point()
        assert (self.tire.contact_point.pos_from(wheel.center) -
                wheel.symbols["r"] * self.int_frame.z).express(wheel.frame).simplify(
        ).xreplace({self.q[1]: 0.123, self.q[2]: 1.234}) == 0
        # sqrt(cos(q2)**2) is not simplified

    def test_toroidal_wheel_on_flat_ground(self, _setup_flat_ground) -> None:
        wheel = ToroidalWheel("wheel")
        wheel.define_objects()
        wheel.define_kinematics()
        self.tire.wheel = wheel
        wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        self.tire._set_pos_contact_point()
        assert (self.tire.contact_point.pos_from(wheel.center) -
                wheel.symbols["r"] * self.int_frame.z + wheel.symbols["tr"] *
                self.ground.get_normal(self.tire.contact_point)).express(
            wheel.frame).simplify().xreplace({self.q[1]: 0.123, self.q[2]: 1.234}) == 0
        # sqrt(cos(q2)**2) is not simplified

    def test_not_implemented_combinations(self) -> None:
        class NewGround(GroundBase):
            def get_normal(self, position):
                return -self.body.z

            def get_tangent_vectors(self, position):
                return (self.frame.x, self.frame.y)

            def set_pos_point(self, point, position) -> None:
                point.set_pos(self.origin, position[0] * self.frame.x +
                              position[1] * self.frame.y)

        class NewWheel(WheelBase):
            @property
            def center(self):
                return self.body.masscenter

            def rotation_axis(self):
                return self.frame.y

        for wheel_cls, ground_cls in [(KnifeEdgeWheel, NewGround),
                                      (NewWheel, FlatGround),
                                      (NewWheel, NewGround)]:
            tire = MyTire("tire")
            tire.ground = ground_cls("ground")
            tire.wheel = wheel_cls("wheel")
            tire.ground.define_objects()
            tire.wheel.define_objects()
            tire.define_objects()
            tire.ground.define_kinematics()
            tire.wheel.define_kinematics()
            with pytest.raises(NotImplementedError):
                tire._set_pos_contact_point()

    def test_upward_radial_axis(self, _setup_flat_ground):
        wheel = KnifeEdgeWheel("wheel")
        wheel.define_objects()
        wheel.define_kinematics()
        self.tire.wheel = wheel
        wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        self.tire.upward_radial_axis = -self.int_frame.z
        self.tire._set_pos_contact_point()
        assert (self.tire.contact_point.pos_from(wheel.center) -
                wheel.symbols["r"] * self.int_frame.z).simplify() == 0

    def test_upward_radial_axis_invalid(self, _setup_flat_ground):
        self.tire.wheel = KnifeEdgeWheel("wheel")
        self.tire.wheel.define_objects()
        self.tire.wheel.define_kinematics()
        self.tire.wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        normal = self.ground.get_normal(self.tire.contact_point)
        with pytest.raises(TypeError):  # no vector
            self.tire.upward_radial_axis = 5
        with pytest.raises(ValueError):  # not normalize
            self.tire.upward_radial_axis = 2 * self.int_frame.z
        with pytest.raises(ValueError):  # not radial
            self.tire.upward_radial_axis = normal
        with pytest.raises(ValueError):  # not correct with respect to normal
            self.tire.upward_radial_axis = cross(
                self.tire.wheel.rotation_axis, normal).normalize()

    @pytest.mark.parametrize("with_wheel", [True, False])
    def test_on_ground_default(self, _setup_flat_ground, with_wheel):
        if with_wheel:
            self.tire.wheel = KnifeEdgeWheel("wheel")
        assert not self.tire.on_ground

    def test_on_ground_unconnected(self, _setup_flat_ground):
        self.tire.wheel = KnifeEdgeWheel("wheel")
        self.tire.wheel.define_objects()
        self.tire.wheel.define_kinematics()
        self.tire.wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        self.tire._set_pos_contact_point()
        assert not self.tire.on_ground

    @pytest.mark.parametrize("off_ground", [True, False])
    def test_on_ground_computation(self, _setup_flat_ground, off_ground):
        self.tire.wheel = KnifeEdgeWheel("wheel")
        self.tire.wheel.define_objects()
        self.tire.wheel.define_kinematics()
        self.tire.wheel.frame.orient_axis(self.int_frame, self.q[2], self.int_frame.y)
        self.tire._set_pos_contact_point()
        self.tire.contact_point.set_pos(
            self.ground.origin,
            int(off_ground) * dynamicsymbols("q3") * self.ground.frame.z)
        assert self.tire.on_ground != off_ground


class TestNonHolonomicTireModel:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.model = create_model_of_connection(NonHolonomicTire)("model")
        self.model.ground = FlatGround("ground")
        self.model.wheel = KnifeEdgeWheel("wheel")
        self.model.conn = NonHolonomicTire("tire_model")

    def test_default(self) -> None:
        self.model.define_connections()
        self.model.define_objects()
        assert self.model.conn.name == "tire_model"
        assert isinstance(self.model.conn.system, System)

    @pytest.mark.parametrize("on_ground", [True, False])
    def test_compute_on_ground(self, on_ground: bool) -> None:
        self.model.define_connections()
        self.model.define_objects()
        self.model.conn.on_ground = on_ground
        ground, wheel, tire_model = (
            self.model.ground, self.model.wheel, self.model.conn)
        t = dynamicsymbols._t
        q1, q2, x, y, z = dynamicsymbols("q1 q2 x y z")
        wheel.frame.orient_body_fixed(ground.frame, (q1, q2, 0), "zyx")
        ground.set_pos_point(tire_model.contact_point, (x, y))
        if not on_ground:
            tire_model.contact_point.set_pos(
                ground.origin, tire_model.contact_point.pos_from(
                    ground.origin) + z * ground.get_normal(tire_model.contact_point))
        self.model.define_kinematics()
        self.model.define_loads()
        self.model.define_constraints()
        fnh = [
            wheel.radius * cos(q1) * q2.diff(t) + x.diff(t),
            wheel.radius * sin(q1) * q2.diff(t) + y.diff(t),
        ]
        assert len(tire_model.system.holonomic_constraints) == int(not on_ground)
        assert len(tire_model.system.nonholonomic_constraints) == 2
        if not on_ground:
            assert (tire_model.system.holonomic_constraints[0] - z
                    ).simplify() == 0
        for fnhi in tire_model.system.nonholonomic_constraints:
            assert (fnhi - fnh[0]).simplify() == 0 or (fnhi - fnh[1]).simplify() == 0
