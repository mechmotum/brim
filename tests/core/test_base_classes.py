from __future__ import annotations

from typing import Any

import pytest
from brim.bicycle import FlatGround, KnifeEdgeWheel, NonHolonomicTire
from brim.core import (
    LoadGroupBase,
    ModelBase,
    ModelRequirement,
    Registry,
    set_default_convention,
)
from brim.other.rolling_disc import RollingDisc
from sympy import S, Symbol
from sympy.physics.mechanics import System, Torque, dynamicsymbols


class MyLoad(LoadGroupBase):
    required_parent_type = KnifeEdgeWheel

    @property
    def descriptions(self) -> dict[Any, str]:
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

    @pytest.fixture()
    def _create_model(self) -> None:
        class MyTire(NonHolonomicTire):
            """Tire with a custom symbol."""

            @property
            def descriptions(self) -> dict[Any, str]:
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
        self.disc.disc = KnifeEdgeWheel("disc")
        self.disc.ground = FlatGround("ground")
        self.disc.tire = MyTire("tire")
        self.load_group = MyLoad("load")
        self.disc.disc.add_load_groups(self.load_group)

    def test_init(self) -> None:
        disc = RollingDisc("model")
        assert isinstance(disc, ModelBase)
        assert str(disc) == "model"
        assert repr(disc) == "RollingDisc('model')"
        assert disc.name == "model"
        assert disc.disc is None
        assert disc.ground is None
        assert disc.tire is None

    @pytest.mark.parametrize("name", ["", " ", "my model", "my,model", "my:model"])
    def test_invalid_name(self, name) -> None:
        with pytest.raises(ValueError):
            RollingDisc(name)

    def test_invalid_model(self) -> None:
        disc = RollingDisc("model")
        with pytest.raises(TypeError):
            disc.disc = FlatGround("ground")

    def test_invalid_connection(self) -> None:
        disc = RollingDisc("model")
        with pytest.raises(TypeError):
            disc.tire = KnifeEdgeWheel("disc")

    def test_overwrite_submodel_of_connection(self, _create_model) -> None:
        self.disc.define_connections()
        self.disc.define_objects()
        self.disc.tire.wheel = KnifeEdgeWheel("disc2")
        assert self.disc.tire.wheel.name == "disc2"

    def test_invalid_submodel_of_connection(self, _create_model) -> None:
        with pytest.raises(TypeError):
            self.disc.tire.ground = KnifeEdgeWheel("disc")

    def test_get_description_own_description(self, _create_model) -> None:
        self.disc.define_all()
        assert (self.disc.disc.get_description(self.disc.disc.radius) ==
                self.disc.disc.descriptions[self.disc.disc.radius])

    def test_get_description_of_submodel(self, _create_model) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.disc.disc.radius) is not None

    def test_get_description_of_connection(self, _create_model) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.disc.tire.symbols["my_sym2"]) is not None

    def test_get_description_of_load_group(self, _create_model) -> None:
        self.disc.define_all()
        assert self.disc.get_description(self.load_group.symbols["T"]) is not None

    def test_get_description_of_not_existing_symbol(self, _create_model) -> None:
        self.disc.define_all()
        assert self.disc.get_description(Symbol("not_existing_symbol")) is None

    def test_traversal_get_all_symbols(self, _create_model) -> None:
        self.disc.define_all()
        assert self.disc.get_all_symbols() == {
            self.disc.disc.symbols["r"], self.disc.tire.symbols["my_sym1"],
            self.disc.tire.symbols["my_sym2"], self.load_group.symbols["T"]
        }
        assert self.disc.disc.get_all_symbols() == {
            self.disc.disc.symbols["r"], self.load_group.symbols["T"]}
        assert self.disc.tire.get_all_symbols() == {
            self.disc.disc.symbols["r"], self.disc.tire.symbols["my_sym1"],
            self.disc.tire.symbols["my_sym2"], self.load_group.symbols["T"]}
        assert self.disc.disc.load_groups[0].get_all_symbols() == {
            self.load_group.symbols["T"]}

    @pytest.mark.parametrize("sym, expected", [
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
    def test_type_get_all_symbols(self, sym, expected, _create_model) -> None:
        self.disc.define_connections()
        self.disc.define_objects()
        self.disc.disc.symbols["r"] = sym
        assert self.disc.get_all_symbols() == {
            self.disc.tire.symbols["my_sym1"], self.disc.tire.symbols["my_sym2"],
            self.load_group.symbols["T"], *expected}

    def test_call_system(self, _create_model) -> None:
        self.disc.define_all()
        assert isinstance(self.disc.system, System)

    def test_load_group_default(self, _create_model) -> None:
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

    def test_to_system(self, _create_model) -> None:
        self.disc.define_all()
        system = self.disc.to_system()
        assert system.loads == (
            Torque(self.disc.disc.frame,
                   self.load_group.symbols["T"] * self.disc.disc.rotation_axis),)

    def test_is_root(self, _create_model) -> None:
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
        for obj in (root, self.disc, self.disc.disc, self.disc.ground):
            assert obj.is_root is None
        root.define_connections()
        root.define_objects()
        assert root.is_root
        for obj in (self.disc, self.disc.disc, self.disc.ground):
            assert not obj.is_root
        root.define_kinematics()
        root.define_loads()
        root.define_constraints()


class TestFromConvention:
    @pytest.fixture(autouse=True)
    def _setup_registry(self, request) -> None:
        def activate_registry():
            old_reg.activate()

        old_reg = Registry()
        old_reg.deactivate()
        request.addfinalizer(activate_registry)

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
