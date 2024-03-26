from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from brim import (
    FlatGround,
    KnifeEdgeWheel,
    NonHolonomicTire,
    RigidFrontFrame,
    RigidRearFrame,
)
from brim.bicycle import MasslessCranks, WhippleBicycle, WhippleBicycleMoore
from brim.utilities.testing import ignore_point_warnings
from sympy import Matrix, Symbol, count_ops, lambdify, linear_eq_to_matrix
from sympy.physics.mechanics import dynamicsymbols, msubs

if TYPE_CHECKING:
    from sympy import Basic


class TestWhippleBicycle:
    def test_default(self) -> None:
        front = WhippleBicycle("bike")
        assert isinstance(front, WhippleBicycleMoore)

    @pytest.mark.parametrize("convention_name, expected_class", [
        ("moore", WhippleBicycleMoore),
    ])
    def test_init(self, convention_name, expected_class) -> None:
        front = WhippleBicycle.from_convention(convention_name, "bike")
        assert isinstance(front, expected_class)

    def test_init_error(self) -> None:
        with pytest.raises(ValueError):
            WhippleBicycle.from_convention("not_implemented", "bike")


class TestWhippleBicycleMoore:
    @staticmethod
    def _get_basu_mandal_values(bike: WhippleBicycleMoore
                                ) -> tuple[dict[Basic, float], dict[Basic, float]]:
        def get_inertia_matrix(model):
            return model.body.central_inertia.to_matrix(model.body.frame)

        constants = {
            bike.front_wheel.symbols["r"]: 0.35,
            bike.rear_wheel.symbols["r"]: 0.3,
            bike.rear_frame.symbols["d1"]: 0.9534570696121849,
            bike.front_frame.symbols["d2"]: 0.2676445084476887,
            bike.front_frame.symbols["d3"]: 0.03207142672761929,
            bike.rear_frame.symbols["l1"]: 0.4707271515135145,
            bike.rear_frame.symbols["l2"]: -0.47792881146460797,
            bike.front_frame.symbols["l3"]: -0.00597083392418685,
            bike.front_frame.symbols["l4"]: -0.3699518200282974,
            bike.rear_frame.body.mass: 85.0,
            bike.rear_wheel.body.mass: 2.0,
            bike.front_frame.body.mass: 4.0,
            bike.front_wheel.body.mass: 3.0,
            get_inertia_matrix(bike.rear_wheel)[0, 0]: 0.0603,
            get_inertia_matrix(bike.rear_wheel)[1, 1]: 0.12,
            get_inertia_matrix(bike.front_wheel)[0, 0]: 0.1405,
            get_inertia_matrix(bike.front_wheel)[1, 1]: 0.28,
            get_inertia_matrix(bike.rear_frame)[0, 0]: 7.178169776497895,
            get_inertia_matrix(bike.rear_frame)[1, 1]: 11.0,
            get_inertia_matrix(bike.rear_frame)[0, 2]: 3.8225535938357873,
            get_inertia_matrix(bike.rear_frame)[2, 2]: 4.821830223502103,
            get_inertia_matrix(bike.front_frame)[0, 0]: 0.05841337700152972,
            get_inertia_matrix(bike.front_frame)[1, 1]: 0.06,
            get_inertia_matrix(bike.front_frame)[0, 2]: 0.009119225261946298,
            get_inertia_matrix(bike.front_frame)[2, 2]: 0.007586622998470264,
            Symbol("g"): 9.81}
        initial_state = {
            **dict(zip(bike.q, (
                -0.0, -0.17447337661787718, -0.0, 0.6206670416476966,
                0.3300446174593725, -0.0, -0.2311385135743, -0.0))),
            **dict(zip(bike.u, (
                2.6703213326046784, -2.453592884421596e-14, -0.7830033527065,
                -0.6068425835418, 0.0119185528069, -8.912989661489, -0.4859824687093,
                -8.0133620584155)))}
        return constants, initial_state

    @pytest.fixture()
    def _setup_default(self) -> None:
        self.bike = WhippleBicycleMoore("bike")
        self.bike.ground = FlatGround("ground")
        self.bike.rear_frame = RigidRearFrame("rear_frame")
        self.bike.front_frame = RigidFrontFrame("front_frame")
        self.bike.rear_wheel = KnifeEdgeWheel("rear_wheel")
        self.bike.front_wheel = KnifeEdgeWheel("front_wheel")
        self.bike.rear_tire = NonHolonomicTire("rear_tire")
        self.bike.front_tire = NonHolonomicTire("front_tire")

    def test_basu_mandal(self, _setup_default) -> None:
        t = dynamicsymbols._t
        self.bike.define_all()
        system = self.bike.to_system()
        system.apply_uniform_gravity(-Symbol("g") * self.bike.ground.get_normal(
            self.bike.ground.origin))
        system.q_ind = [*self.bike.q[:4], *self.bike.q[5:]]
        system.q_dep = [self.bike.q[4]]
        system.u_ind = [self.bike.u[3], *self.bike.u[5:7]]
        system.u_dep = [*self.bike.u[:3], self.bike.u[4], self.bike.u[7]]
        with ignore_point_warnings():
            system.form_eoms(constraint_solver="CRAMER")

        constants, initial_state = self._get_basu_mandal_values(self.bike)
        p, p_vals = zip(*constants.items())
        q0 = [initial_state[qi] for qi in system.q]
        u0 = [initial_state[ui] for ui in system.u]
        eval_sys = lambdify((system.q, system.u, p),
                            (system.mass_matrix, system.forcing), cse=True)
        eval_fnh = lambdify((system.q, system.u, p),
                            system.nonholonomic_constraints.xreplace(
                                system.eom_method.kindiffdict()), cse=True)
        assert np.allclose(eval_fnh(q0, u0, p_vals).ravel(), np.zeros(4))
        md, gd = eval_sys(q0, u0, p_vals)
        ud0 = np.linalg.solve(md.astype(np.float64), gd.astype(np.float64)).ravel()
        expected_state = dict(zip(self.bike.u.diff(t), (
                0.5903429412631302, -2.090870556233152, -0.8353281706376822,
                7.855528112824374, -0.12055438978863461, -1.8472554144218631,
                4.6198904039391895, -2.4548072904552343)))
        ud0_expected = [expected_state[udi] for udi in system.u.diff(t)]
        assert np.allclose(ud0, ud0_expected)

    def test_descriptions(self, _setup_default) -> None:
        self.bike.define_connections()
        self.bike.define_objects()
        for qi in self.bike.q:
            assert self.bike.descriptions[qi]
        for ui in self.bike.u:
            assert self.bike.descriptions[ui]

    def test_cranks(self, _setup_default) -> None:
        self.bike.cranks = MasslessCranks("cranks")
        self.bike.define_all()
        assert self.bike.get_description(self.bike.symbols["gear_ratio"]) is not None
        rf = self.bike.rear_frame.wheel_hub.frame
        assert self.bike.cranks.center_point.pos_from(self.bike.rear_wheel.center).diff(
            dynamicsymbols._t, rf) == 0
        assert (self.bike.cranks.frame.ang_vel_in(rf).dot(rf.y) ==
                self.bike.u[7] / self.bike.symbols["gear_ratio"])

    @pytest.mark.parametrize("normal, upward, longitudinal, lateral", [
        ("+z", Matrix([0, 0, 1]), Matrix([-1, 0, 0]), Matrix([0, 1, 0])),
        ("-z", Matrix([0, 0, -1]), Matrix([1, 0, 0]), Matrix([0, 1, 0]))])
    def test_define_tire_axes(self, _setup_default, normal, upward, longitudinal,
                              lateral) -> None:
        self.bike.ground = FlatGround("ground", normal=normal)
        self.bike.define_all()
        tire = self.bike.rear_tire
        assert count_ops(tire.upward_radial_axis.to_matrix(self.bike.ground.frame)) < 30
        assert count_ops(tire.longitudinal_axis.to_matrix(self.bike.ground.frame)) < 30
        assert count_ops(tire.lateral_axis.to_matrix(self.bike.ground.frame)) < 30

    @pytest.mark.parametrize("normal", ["+x", "-x", "+y", "-y"])
    def test_do_not_define_upward_radial_axis(self, _setup_default, normal) -> None:
        self.bike.ground = FlatGround("ground", normal=normal)
        self.bike.define_all()
        tire = self.bike.rear_tire
        assert count_ops(tire.upward_radial_axis.to_matrix(self.bike.ground.frame)) > 30
        assert count_ops(tire.longitudinal_axis.to_matrix(self.bike.ground.frame)) > 30
        assert count_ops(tire.lateral_axis.to_matrix(self.bike.ground.frame)) > 30

    @pytest.mark.parametrize("compute_rear", [True, False])
    @pytest.mark.parametrize("compute_front", [True, False])
    def test_computation_normal_force_nominal_config(self, _setup_default, compute_rear,
                                                     compute_front) -> None:
        if not compute_rear and not compute_front:
            return
        self.bike.rear_tire.compute_normal_force = compute_rear
        self.bike.front_tire.compute_normal_force = compute_front
        self.bike.define_all()
        system = self.bike.to_system()
        system.apply_uniform_gravity(-Symbol("g") * self.bike.ground.get_normal(
            self.bike.ground.origin))
        system.q_ind = [*self.bike.q[:4], *self.bike.q[5:]]
        system.q_dep = [self.bike.q[4]]
        system.u_ind = [self.bike.u[3], *self.bike.u[5:7]]
        system.u_dep = [*self.bike.u[:3], self.bike.u[4], self.bike.u[7]]
        assert len(system.u_aux) == int(compute_rear) + int(compute_front)
        with ignore_point_warnings():
            system.form_eoms(constraint_solver="CRAMER")
        constants, _ = self._get_basu_mandal_values(self.bike)
        aux_eqs = system.eom_method.auxiliary_eqs
        fn_syms = []
        if compute_rear:
            fn_syms.extend([self.bike.rear_tire.symbols["Fz"]])
        if compute_front:
            fn_syms.extend([self.bike.front_tire.symbols["Fz"]])
        fn_eqs = Matrix.cramer_solve(*linear_eq_to_matrix(aux_eqs, fn_syms))
        zero = 1e-10
        zero_config = {ui.diff(): zero for ui in system.u}
        zero_config.update({ui: zero for ui in system.u})
        zero_config.update({qi: zero for qi in system.q})
        zero_config[self.bike.q[4]] = np.pi / 10
        fn_vals = [float(val) for val in msubs(fn_eqs, zero_config, constants)]
        if compute_rear:
            np.testing.assert_allclose([fn_vals[0]], [612.836470588236])
        if compute_front:
            np.testing.assert_allclose([fn_vals[-1]], [309.303529411765])
