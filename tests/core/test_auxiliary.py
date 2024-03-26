import pytest
from brim.core.auxiliary import AuxiliaryData, AuxiliaryDataHandler
from sympy import simplify, solve, sqrt, symbols
from sympy.physics.mechanics import (
    Force,
    Particle,
    Point,
    ReferenceFrame,
    System,
    dynamicsymbols,
)

frame = ReferenceFrame("frame")
point = Point("point")
uaux = dynamicsymbols("uaux")
faux = dynamicsymbols("faux")

class AuxiliarySetup:
    @pytest.fixture()
    def _setup_cart_pendulum(self) -> None:
        """Cart with a two link pendulum."""
        self.x, self.v = dynamicsymbols("x v")
        self.q1, self.q2, self.u1, self.u2 = dynamicsymbols("q1:3 u1:3")
        self.l = symbols("l:3")
        self.uay, self.fay, self.ual, self.fal = dynamicsymbols("uay fay ual fal")
        self.inertial_frame = ReferenceFrame("inertial_frame")
        self.inertial_point = Point("inertial_point")
        self.inertial_point.set_vel(self.inertial_frame, 0)
        self.f1 = ReferenceFrame("link1")
        self.f2 = ReferenceFrame("link2")
        self.f1.orient_axis(self.inertial_frame, self.q1, self.inertial_frame.z)
        self.f1.set_ang_vel(self.inertial_frame, self.u1 * self.inertial_frame.z)
        self.f2.orient_axis(self.f1, self.q2, self.f1.z)
        self.f2.set_ang_vel(self.f1, self.u2 * self.f1.z)
        self.cart = self.inertial_point.locatenew(
            "cart", self.x * self.inertial_frame.x)
        self.cart.set_vel(self.inertial_frame, self.v * self.inertial_frame.x)
        self.p1 = self.cart.locatenew("p1", -self.l[0] * self.f1.y)
        self.p2 = self.p1.locatenew("p2", -self.l[1] * self.f2.y)
        # Additional point attached to the first link.
        self.p3 = self.p1.locatenew("p3", self.l[2] * self.f1.x)
        self.p1.v2pt_theory(self.cart, self.inertial_frame, self.f1)
        # Don't define the velocity of p2, p3 to possibly cause errors.

        self.ld_fz = AuxiliaryData(
            self.cart, self.inertial_frame.y, self.uay, self.fay)
        self.ld_j2 = AuxiliaryData(self.p1, -self.f2.y, self.ual, self.fal)
        self.noncontributing_loads = [self.ld_fz, self.ld_j2]

class TestAuxiliaryData(AuxiliarySetup):
    @pytest.mark.parametrize("args, kwargs, expected", [
        ((point, frame.x, uaux, faux), {}, (point, frame.x, uaux, faux)),
        ((), {"location": point, "direction": frame.z, "speed_symbol": uaux,
              "load_symbol": faux}, (point, frame.z, uaux, faux)),
    ])
    def test_init_force(self, args, kwargs, expected) -> None:
        force = AuxiliaryData(*args, **kwargs)
        assert force.location == expected[0]
        assert force.direction == expected[1]
        assert force.speed_symbol == expected[2]
        assert force.load_symbol == expected[3]

    def test_init_torque(self, _setup_cart_pendulum) -> None:
        with pytest.raises(NotImplementedError):
            AuxiliaryData(self.f1, self.inertial_frame.x, self.uay, self.fay)

    @pytest.mark.parametrize("args", [
        (frame.x, frame.y, uaux, faux),
        (point, 5, uaux, faux),
    ])
    def test_init_invalid_type(self, args):
        with pytest.raises(TypeError):
            AuxiliaryData(*args)

    def test_is_force_torque(self, _setup_cart_pendulum) -> None:
        assert self.ld_fz.is_force
        assert not self.ld_fz.is_torque

    def test_get_load_force(self, _setup_cart_pendulum) -> None:
        force = self.ld_fz.get_load(self.inertial_frame)
        pytest.raises(ValueError, lambda: force.point.pos_from(self.inertial_point))
        assert isinstance(force, Force)
        assert force.point.vel(self.inertial_frame) == self.uay * self.inertial_frame.y
        assert force.point.acc(self.inertial_frame) == (
            self.uay.diff() * self.inertial_frame.y)
        assert force.force == self.fay * self.inertial_frame.y

    def test_auxiliary_velocity(self, _setup_cart_pendulum) -> None:
        assert self.ld_fz.auxiliary_velocity == self.uay * self.inertial_frame.y
        assert self.ld_j2.auxiliary_velocity == -self.ual * self.f2.y


class TestAuxiliaryHandler(AuxiliarySetup):
    @pytest.fixture()
    def _setup_handler(self, _setup_cart_pendulum) -> None:
        self.handler = AuxiliaryDataHandler(self.inertial_frame, self.inertial_point)
        self.handler.auxiliary_data_list.extend(self.noncontributing_loads)

    @pytest.mark.parametrize("args, kwargs, exp_frame, exp_point", [
        ((frame, point), {}, frame, point),
        ((), {"inertial_frame": frame, "inertial_point": point}, frame, point),
    ])
    def test_init(self, args, kwargs, exp_frame, exp_point) -> None:
        handler = AuxiliaryDataHandler(*args, **kwargs)
        assert handler.inertial_frame == exp_frame
        assert handler.inertial_point == exp_point
        assert handler.auxiliary_data_list == []
        assert handler.auxiliary_forces_data == ()
        assert handler.auxiliary_torques_data == ()

    @pytest.mark.parametrize("args", [(frame, 5), (5, point)])
    def test_init_invalid_type(self, args) -> None:
        with pytest.raises(TypeError):
            AuxiliaryDataHandler(*args)

    def test_from_system(self, _setup_handler) -> None:
        sys = System()
        handler = AuxiliaryDataHandler.from_system(sys)
        assert handler.inertial_frame == sys.frame
        assert handler.inertial_point == sys.fixed_point

    @pytest.mark.parametrize("get_childs", [
        "_pos_dict", lambda pt: pt._pos_dict.keys()])
    def test_create_tree_points(self, _setup_handler, get_childs) -> None:
        tree = AuxiliaryDataHandler._extract_tree(self.inertial_point, get_childs)
        expected_tree = {
            self.inertial_point: [self.cart],
            self.cart: [self.p1],
            self.p1: [self.p2, self.p3],
            self.p2: [],
            self.p3: [],
        }
        assert set(tree) == set(expected_tree)
        for p, chlds in tree.items():
            assert set(chlds) == set(expected_tree[p])
            assert len(chlds) == len(expected_tree[p])

    @pytest.mark.parametrize("graph, root, tree_exp", [
        ({1: [2, 3], 2: [1, 4], 3: [1], 4: [2]}, 1, {1: [2, 3], 2: [4], 3: [], 4: []}),
        ({1: [2, 3], 2: [1, 4], 3: [1], 4: [2]}, 2, {2: [1, 4], 1: [3], 4: [], 3: []}),
    ])
    def test_extract_tree(self, graph, root, tree_exp) -> None:
        get_childs = lambda pt: graph[pt]  # noqa: E731
        tree = AuxiliaryDataHandler._extract_tree(root, get_childs)
        assert tree == tree_exp

    @pytest.mark.parametrize("graph, root", [
        ({1: [2, 3], 2: [1, 3], 3: [1, 2]}, 1),
        ({1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}, 1),
        ({1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}, 2),
    ])
    def test_extract_tree_cycle(self, graph, root) -> None:
        get_childs = lambda pt: graph[pt]  # noqa: E731
        with pytest.raises(ValueError):
            AuxiliaryDataHandler._extract_tree(root, get_childs)

    @pytest.mark.parametrize("point, parent", [
        ("inertial_point", None), ("cart", "inertial_point"), ("p2", "p1"),
        (point, None)])
    def test_get_parent(self, _setup_handler, point, parent) -> None:
        point = getattr(self, point) if isinstance(point, str) else point
        parent = getattr(self, parent) if isinstance(parent, str) else parent
        self.handler.retrieve_graphs()
        assert self.handler._get_parent(point) == parent

    def test_compute_velocity_inertial_point(self) -> None:
        handler = AuxiliaryDataHandler(frame, point)
        point2 = point.locatenew("point2", 5 * frame.x)
        point2.set_vel(frame, 10 * frame.y)
        assert handler._compute_velocity(point, frame) == 10 * frame.y

    @pytest.mark.parametrize("pass_parent", [True, False])
    @pytest.mark.parametrize("point, parent, vel", [
        ("p2", "p1", "u2 * f1.x"),  # simple retrieval
        ("p3", "p2", "p3.v2pt_theory(p2, f1, f2)"),  # v2pt_theory
        ("p3", "p2", "u2 * f1.x - u1 * 3 * f2.x"),  # v2pt_theory
        ("p4", "p2", "p4.v1pt_theory(p2, f1, f2)"),  # v1pt_theory
        ("p4", "p2", "u2 * f1.x + u3 * f2.x + u1 * q3 * f2.y"),  # v1pt_theory
        ("p5", "p4", "u2 * f1.x + (u3 - 2 * u1) * f2.x + u1 * q3 * f2.y"),  # derivative
    ])
    def test_compute_velocity(self, pass_parent, point, parent, vel) -> None:
        q1, q2, q3, u1, u2, u3 = dynamicsymbols("q1:4 u1:4")
        q1d, q2d, q3d = dynamicsymbols("q1:4", 1)
        f1 = ReferenceFrame("f1")
        f2 = ReferenceFrame("f2")
        f2.orient_axis(f1, q1, f1.z)
        f2.set_ang_vel(f1, u1 * f1.z)
        p1 = Point("p1")
        p1.set_vel(f1, 0)
        p2 = p1.locatenew("p2", q2 * f1.x)
        p2.set_vel(f1, u2 * f1.x)
        p2.set_vel(f2, 0)
        p3 = p2.locatenew("p3", 3 * f2.y)
        p3.set_vel(f2, 0)
        p4 = p2.locatenew("p4", q3 * f2.x)
        p4.set_vel(f2, u3 * f2.x)
        p5 = p4.locatenew("p5", 2 * f2.y)  # noqa: F841
        handler = AuxiliaryDataHandler(f1, p1)
        point = eval(point) if isinstance(point, str) else point
        vel_exp = eval(vel) if isinstance(vel, str) else vel
        if pass_parent:
            vel = handler._compute_velocity(point, eval(parent))
        else:
            vel = handler._compute_velocity(point)
        assert (vel - vel_exp).express(f2).simplify() == 0

    def test_compute_velocity_disconnected(self) -> None:
        handler = AuxiliaryDataHandler(frame, point)
        point.set_vel(frame, 0)
        point2 = Point("point2")
        with pytest.raises(ValueError):
            handler._compute_velocity(point2)

    def test_apply_speeds(self, _setup_handler) -> None:
        self.handler.apply_speeds()
        cart_vel = self.v * self.inertial_frame.x + self.uay * self.inertial_frame.y
        p1_vel = cart_vel + self.l[0] * self.u1 * self.f1.x - self.ual * self.f2.y
        p2_vel = p1_vel + self.l[1] * (self.u1 + self.u2) * self.f2.x
        p3_vel = p1_vel + self.l[2] * self.u1  * self.f1.y
        assert self.cart.vel(self.inertial_frame) == cart_vel
        assert self.p1.vel(self.inertial_frame) == p1_vel
        assert self.p2.vel(self.inertial_frame) == p2_vel
        assert self.p3.vel(self.inertial_frame) == p3_vel

    def test_apply_speeds_twice(self, _setup_handler) -> None:
        self.handler.apply_speeds()
        pytest.raises(ValueError, lambda: self.handler.apply_speeds())

    def test_apply_speeds_disconnected(self, _setup_handler) -> None:
        self.handler.add_noncontributing_force(point, frame.y, uaux, faux)
        with pytest.raises(ValueError):
            self.handler.apply_speeds()

    def test_create_loads(self, _setup_handler) -> None:
        loads = self.handler.create_loads()
        for ld in loads:
            with pytest.raises(ValueError):
                ld.point.pos_from(self.inertial_point)
        assert loads[0].point.vel(self.inertial_frame) == (
            self.uay * self.inertial_frame.y)
        assert loads[0].force == self.fay * self.inertial_frame.y
        assert loads[1].point.vel(self.inertial_frame) == -self.ual * self.f2.y
        assert loads[1].force == -self.fal * self.f2.y

    @pytest.mark.parametrize("point, vel", [
        ("inertial_point", 0),
        ("cart", "uay * inertial_frame.y"),
        ("p1", "uay * inertial_frame.y - ual * f2.y"),
        ("p2", "uay * inertial_frame.y - ual * f2.y"),
        ("p3", "uay * inertial_frame.y - ual * f2.y"),
    ])
    def test_get_auxiliary_velocity(self, _setup_handler, point, vel) -> None:
        inertial_frame = self.inertial_frame  # noqa: F841
        uay, ual, f1, f2 = self.uay, self.ual, self.f1, self.f2  # noqa: F841
        point = getattr(self, point) if isinstance(point, str) else point
        vel = eval(vel) if isinstance(vel, str) else vel
        self.handler.apply_speeds()
        assert self.handler.get_auxiliary_velocity(point) == vel

    def test_get_auxiliary_velocity_not_applied(self, _setup_handler) -> None:
        with pytest.raises(ValueError):
            self.handler.get_auxiliary_velocity(self.p2)

    def test_get_auxiliary_velocity_disconnected(self, _setup_handler) -> None:
        self.handler.apply_speeds()
        with pytest.raises(ValueError):
            self.handler.get_auxiliary_velocity(Point("point2"))

    def test_sliding_box(self):
        sign = lambda x: x / sqrt(x**2)  # noqa: E731
        # Define the symbols.
        m, g, mu = symbols("m g mu")
        t = dynamicsymbols._t
        x, v, uaux, fext, fn = dynamicsymbols("x v uaux fext fn")

        # Initialize the system object.
        sb_sys = System()

        # Create the box.
        box = Particle("box", Point("box_o"), m)
        box.masscenter.set_pos(sb_sys.fixed_point, x * sb_sys.x)
        box.masscenter.set_vel(sb_sys.frame, (v) * sb_sys.x)
        sb_sys.add_bodies(box)

        # Specify the coordinates and speeds.
        sb_sys.q_ind = [x]
        sb_sys.u_ind = [v]
        sb_sys.kdes = [x.diff(t) - v]
        sb_sys.u_aux = [uaux]

        # Apply loads.
        sb_sys.apply_uniform_gravity(-g * sb_sys.y)
        sb_sys.add_loads(
            Force(box.masscenter, fext * sb_sys.x),  # Extenal force
            Force(box.masscenter, -sign(v) * mu * fn * sb_sys.x),  # Friction force
        )

        # Introduce noncontributing load.
        handler = AuxiliaryDataHandler(sb_sys.frame, sb_sys.fixed_point)
        handler.add_noncontributing_force(box.masscenter, sb_sys.y, uaux, fn)
        handler.apply_speeds()

        sb_sys.add_loads(*handler.create_loads())

        # Form the equations of motion.
        sb_sys.validate_system()
        eoms = sb_sys.form_eoms()

        # Test the output.
        aux_eqs = sb_sys.eom_method.auxiliary_eqs
        fn_eq = solve(aux_eqs, fn)[fn]
        assert fn_eq == g * m
        assert simplify((solve(eoms, v.diff(t))[v.diff(t)] -
                        (-mu * fn * sign(v) + fext)/m).xreplace({fn: fn_eq})) == 0
