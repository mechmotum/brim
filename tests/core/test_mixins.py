from __future__ import annotations

import pytest
from brim.core.base_classes import ModelBase
from brim.core.mixins import NewtonianBodyMixin
from sympy.physics.mechanics import RigidBody

try:
    from brim.utilities.plotting import PlotModel
    from symmeplot.matplotlib import PlotBody
except ImportError:
    PlotModel = None


class MyModel(NewtonianBodyMixin, ModelBase):
    pass


class TestNewtonianBodyMixin:
    def test_mixin(self) -> None:
        model = MyModel("name")
        model.define_all()
        assert isinstance(model.body, RigidBody)
        assert model.system.bodies[0] is model.body
        assert model.frame is model.body.frame
        assert model.x is model.frame.x
        assert model.y is model.frame.y
        assert model.z is model.frame.z
        assert model.system.fixed_point is model.body.masscenter
        assert model.body.masscenter.vel(model.frame) == 0

    @pytest.mark.skipif(PlotModel is None, reason="symmeplot not installed")
    def test_plotting(self) -> None:
        model = MyModel("name")
        model.define_all()
        plot_model = PlotModel(model.system.frame, model.system.fixed_point, model)
        assert len(plot_model.children) == 1
        assert isinstance(plot_model.children[0], PlotBody)
