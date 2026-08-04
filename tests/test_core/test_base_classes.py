from __future__ import annotations

import pytest
from sympy import S, Symbol
from sympy.physics.mechanics import (
    Particle,
    Point,
    ReferenceFrame,
    System,
    Torque,
    dynamicsymbols,
)

from symbrim.bicycle import FlatGround, KnifeEdgeWheel, NonHolonomicTire, WhippleBicycle
from symbrim.core import (
    LoadGroupBase,
    ModelBase,
    ModelRequirement,
    Registry,
    set_default_convention,
)
from symbrim.other.rolling_disc import RollingDisc


class MyLoad(LoadGroupBase):
    required_parent_type = KnifeEdgeWheel

    @property
    def descriptions(self) -> dict[object, str]:
        """Dictionary of descriptions of the load group's attributes."""
        return {
            **super().descriptions,
            self.symbols["T"]: "Torque applied to the disc.",
        }

    def _define_objects(self) -> None:
        """Define the objects in the load group."""
        self.symbols["T"] = Symbol(self._add_prefix("T"))

    def _define_loads(self) -> None:
        """Define the loads in the load group."""
        self.system.add_loads(Torque(
            self.parent.frame, self.symbols["T"] * self.parent.rotation_axis))


class TestModelBase:
    """Test the ModelBase class.

    Explanation
    -----------
    As ModelBase is an abstract class, this test actually uses the rolling disc to test
    certain characteristics of the ModelBase class.
    """

    @pytest.fixture
    def _create_model(self) -> None:
        class MyTire(NonHolonomicTire):
            """Tire with a custom symbol."""

            @property
            def descriptions(self) -> dict[object, str]:
                """Dictionary of descriptions of the tire's attributes."""
                return {
                    **super().descriptions,
                    self.symbols["my_sym1"]: "My symbol.",
                    self.symbols["my_sym2"]: "My symbol.",
                }

            def _define_objects(self) -> None:
                """Define the objects in the tire."""
                super()._define_objects()
                self.symbols["my_sym1"] = Symbol(self._add_prefix("my_sym1"))
                self.symbols["my_sym2"] = Symbol(self._add_prefix("my_sym2"))

        self.disc = RollingDisc("rolling_disc")
        self.disc.wheel = KnifeEdgeWheel("disc")
        self.disc.ground = FlatGround("ground")
        self.disc.tire = MyTire("tire")
        self.load_group = MyLoad("load")
        self.disc.wheel.add_load_groups(self.load_group)

    def test_init(self) -> None:
        disc = RollingDisc("model")
        assert isinstance(disc, ModelBase)
        assert str(disc) == "model"
        assert repr(disc) == "RollingDisc('model')"
        assert disc.name == "model"
        assert disc.wheel is None
        assert disc.ground is None
        assert disc.tire is None

    @pytest.mark.parametrize("name", ["", " ", "my model", "my,model", "my:model"])
    def test_invalid_name(self, name) -> None:
        with pytest.raises(ValueError):
            RollingDisc(name)

    def test_invalid_model(self) -> None:
        disc = RollingDisc("model")
        with pytest.raises(TypeError):
            disc.wheel = FlatGround("ground")

    def test_invalid_connection(self) -> None:
        disc = RollingDisc("model")
        with pytest.raises(TypeError):
            disc.tire = KnifeEdgeWheel("disc")

    @pytest.mark.usefixtures("_create_model")
    def test_overwrite_submodel_of_connection(self) -> None:
        self.disc.define_connections()
        self.disc.define_objects()
        self.disc.tire.wheel = KnifeEdgeWheel("disc2")
        assert self.disc.tire.wheel.name == "disc2"

    @pytest.mark.usefixtures("_create_model")
    def test_invalid_submodel_of_connection(self) -> None:
        with pytest.raises(TypeError):
            self.disc.tire.ground = KnifeEdgeWheel("disc")

    @pytest.mark.usefixtures("_create_model")
    def test_get_description_own_description(self) -> None:
        self.disc.define_all()
        assert (self.disc.wheel.get_description(self.disc.wheel.radius) ==
                self.disc.wheel.descriptions[self.disc.wheel.radius])

    @pytest.mark.usefixtures("_create_model")
    def test_get_description_of_submodel(self) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.disc.wheel.radius) is not None

    @pytest.mark.usefixtures("_create_model")
    def test_get_description_of_connection(self) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.disc.tire.symbols["my_sym2"]) is not None

    @pytest.mark.usefixtures("_create_model")
    def test_get_description_of_load_group(self) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.load_group.symbols["T"]) is not None

    @pytest.mark.usefixtures("_create_model")
    def test_get_description_of_not_existing_symbol(self) -> None:
        self.disc.define_all()
        assert self.disc.get_description(Symbol("not_existing_symbol")) is None

    @pytest.mark.usefixtures("_create_model")
    def test_traversal_get_all_symbols(self) -> None:
        self.disc.define_all()
        wheel_body_symbols = {
            self.disc.wheel.body.mass,
            *{
                symbol
                for entry in self.disc.wheel.body.central_inertia.to_matrix(
                    self.disc.wheel.body.frame
                )
                for symbol in entry.free_symbols
            },
        }
        assert self.disc.get_all_symbols() == {
            self.disc.wheel.symbols["r"], self.disc.tire.symbols["my_sym1"],
            self.disc.tire.symbols["my_sym2"], self.load_group.symbols["T"],
            *wheel_body_symbols,
        }
        assert self.disc.wheel.get_all_symbols() == {
            self.disc.wheel.symbols["r"], self.load_group.symbols["T"],
            *wheel_body_symbols}
        assert self.disc.tire.get_all_symbols() == {
            self.disc.tire.symbols["my_sym1"], self.disc.tire.symbols["my_sym2"]}
        assert self.disc.wheel.load_groups[0].get_all_symbols() == {
            self.load_group.symbols["T"]}

    @pytest.mark.parametrize(("sym", "expected"), [
        (Symbol("r_my"), {Symbol("r_my")}),
        (dynamicsymbols("r_var"), {dynamicsymbols("r_var")}),
        (dynamicsymbols("r_var", 1),
         {dynamicsymbols("r_var"), dynamicsymbols("r_var", 1)}),
        (Symbol("r_def") - dynamicsymbols("r_var"),
         {Symbol("r_def"), dynamicsymbols("r_var")}),
        (0, set()),
        (1.0, set()),
        (S.Zero, set()),
    ])
    @pytest.mark.usefixtures("_create_model")
    def test_type_get_all_symbols(self, sym, expected) -> None:
        self.disc.define_connections()
        self.disc.define_objects()
        self.disc.wheel.symbols["r"] = sym
        wheel_body_symbols = {
            self.disc.wheel.body.mass,
            *{
                symbol
                for entry in self.disc.wheel.body.central_inertia.to_matrix(
                    self.disc.wheel.body.frame
                )
                for symbol in entry.free_symbols
            },
        }
        assert self.disc.get_all_symbols() == {
            self.disc.tire.symbols["my_sym1"], self.disc.tire.symbols["my_sym2"],
            self.load_group.symbols["T"], *wheel_body_symbols, *expected}

    def test_particle_body_get_all_symbols(self) -> None:
        class ParticleModel(ModelBase):
            def _define_objects(self) -> None:
                super()._define_objects()
                self._body = Particle("particle", Point("P"))
                self.symbols["m"] = Symbol(self._add_prefix("m"))
                self.body.mass = self.symbols["m"]
                self._system = System(ReferenceFrame("N"), Point("O"))
                self._system.add_bodies(self.body)

            @property
            def body(self) -> Particle:
                return self._body

        model = ParticleModel("particle_model")
        model.define_objects()
        assert model.get_all_symbols() == {model.symbols["m"]}

    @pytest.mark.usefixtures("_create_model")
    def test_call_system(self) -> None:
        self.disc.define_all()
        assert isinstance(self.disc.system, System)

    @pytest.mark.usefixtures("_create_model")
    def test_load_group_default(self) -> None:
        class MyEmptyLoad(LoadGroupBase):
            def __init__(self, name):
                super().__init__(name)
                self.called_objects = False
                self.called_kinematics = False
                self.called_loads = False
                self.called_constraints = False

            def _define_objects(self) -> None:
                self.called_objects = True

            def _define_kinematics(self) -> None:
                self.called_kinematics = True

            def _define_loads(self) -> None:
                self.called_loads = True

            def _define_constraints(self) -> None:
                self.called_constraints = True

        load_model = MyEmptyLoad("load1")
        load_conn = MyEmptyLoad("load2")
        self.disc.add_load_groups(load_model)
        self.disc.tire.add_load_groups(load_conn)
        self.disc.define_all()
        assert load_model.called_objects
        assert load_model.called_kinematics
        assert load_model.called_loads
        assert load_model.called_constraints
        assert load_conn.called_objects
        assert load_conn.called_kinematics
        assert load_conn.called_loads
        assert load_conn.called_constraints

    def test_load_group_undefined_parent(self) -> None:
        load = MyLoad("load")
        assert load.parent is None
        assert load.system is None

    def test_load_group_invalid_parent(self) -> None:
        with pytest.raises(TypeError):
            FlatGround("ground").add_load_groups(MyLoad("load"))

    def test_load_grip_used_parent(self) -> None:
        wheel1 = KnifeEdgeWheel("wheel1")
        wheel2 = KnifeEdgeWheel("wheel2")
        load_group = MyLoad("load")
        wheel1.add_load_groups(load_group)
        with pytest.raises(ValueError):
            wheel2.add_load_groups(load_group)

    @pytest.mark.usefixtures("_create_model")
    def test_to_system(self) -> None:
        self.disc.define_all()
        system = self.disc.to_system()
        assert system.loads == (
            Torque(self.disc.wheel.frame,
                   self.load_group.symbols["T"] * self.disc.wheel.rotation_axis),)

    @pytest.mark.usefixtures("_create_model")
    def test_is_root(self) -> None:
        class MyModel(ModelBase):
            required_models: tuple[ModelRequirement, ...] = (
                ModelRequirement("rolling_disc", RollingDisc, "Rolling disc model."),
            )

            def _define_objects(self) -> None:
                super()._define_objects()
                self._system = System(self.rolling_disc.system.frame,
                                      self.rolling_disc.system.fixed_point)

        root = MyModel("root")
        root.rolling_disc = self.disc
        for obj in (root, self.disc, self.disc.wheel, self.disc.ground):
            assert obj.is_root is None
        root.define_connections()
        root.define_objects()
        assert root.is_root
        for obj in (self.disc, self.disc.wheel, self.disc.ground):
            assert not obj.is_root
        root.define_kinematics()
        root.define_loads()
        root.define_constraints()

    @pytest.mark.parametrize(
        ("ground", "wheel", "tire", "expected"),
        [
            (None, None, None, ("ground", "wheel", "tire")),
            (FlatGround("ground"), None, None, ("wheel", "tire")),
            (
                FlatGround("ground"),
                KnifeEdgeWheel("wheel"),
                NonHolonomicTire("tire"),
                (),
            ),
        ],
        ids=["all_unspecified", "partially_specified", "all_specified"],
    )
    def test_get_unspecified_components(self, ground, wheel, tire, expected) -> None:
        disc = RollingDisc("disc")
        if ground is not None:
            disc.ground = ground
        if wheel is not None:
            disc.wheel = wheel
        if tire is not None:
            disc.tire = tire
        unspecified = disc.get_unspecified_components()
        assert unspecified == expected

    def test_get_unspecified_components_with_optional(self) -> None:
        bike = WhippleBicycle("bike")
        unspecified_hard = bike.get_unspecified_components()
        assert "cranks" not in unspecified_hard
        unspecified_all = bike.get_unspecified_components(optional=True)
        assert "cranks" in unspecified_all

    def test_get_unspecified_components_detailed(self) -> None:
        disc = RollingDisc("disc")
        disc.ground = FlatGround("ground")
        unspecified = disc.get_unspecified_components(detailed=True)
        assert len(unspecified) == 2
        assert all(hasattr(req, "attribute_name") for req in unspecified)
        assert unspecified[0].attribute_name == "wheel"
        assert unspecified[1].attribute_name == "tire"

class TestFromConvention:
    @pytest.fixture(autouse=True)
    def _setup_registry(self, request):
        def activate_registry():
            old_reg.activate()

        old_reg = Registry()
        old_reg.deactivate()
        request.addfinalizer(activate_registry)  # noqa: PT021

        @set_default_convention("default_convention")
        class MyModel(ModelBase):
            pass

        class MyModel2(MyModel):
            convention = "default_convention"

        class MyModel3(MyModel):
            convention = "other_convention"

        class MyModel4(MyModel):
            convention = "double_convention"

        class MyModel5(MyModel):
            convention = "double_convention"

        self.MyModel, self.MyModel2, self.MyModel3, self.MyModel4, self.MyModel5 = (
            MyModel, MyModel2, MyModel3, MyModel4, MyModel5)

    def test_convention_string(self) -> None:
        assert self.MyModel.convention == ""
        assert self.MyModel2.convention == "default_convention"
        assert self.MyModel3.convention == "other_convention"

    def test_default_convention(self) -> None:
        model = self.MyModel("model")
        assert model.convention == "default_convention"
        assert isinstance(model, self.MyModel2)

    def test_convention_argument(self) -> None:
        model = self.MyModel.from_convention("other_convention", "model")
        assert model.convention == "other_convention"
        assert isinstance(model, self.MyModel3)

    def test_not_existing_convention(self) -> None:
        with pytest.raises(ValueError):
            self.MyModel.from_convention("not existing convention", "model")

    def test_double_convention(self) -> None:
        with pytest.raises(ValueError):
            self.MyModel.from_convention("double_convention", "model")

    def test_direct_instantiation(self) -> None:
        model = self.MyModel4("model")
        assert model.convention == "double_convention"
        assert isinstance(model, self.MyModel4)

    def test_set_default_convention_invalid_type(self) -> None:
        with pytest.raises(TypeError):
            @set_default_convention("my_convention")
            class A:
                pass
