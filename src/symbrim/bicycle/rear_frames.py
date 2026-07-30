"""Module containing the models of the rear frame of a bicycle."""
from __future__ import annotations

import contextlib
from abc import abstractmethod
from typing import TYPE_CHECKING

from sympy import Symbol, symbols
from sympy.physics.mechanics import Point, RigidBody, System, inertia

from symbrim.core import Attachment, Hub, ModelBase, set_default_convention

with contextlib.suppress(ImportError):
    import numpy as np
    from bicycleparameters.io import remove_uncertainties
    from bicycleparameters.main import calculate_benchmark_from_measured
    from bicycleparameters.rider import yeadon_vec_to_bicycle_vec
    from dtk.bicycle import benchmark_to_moore

    from symbrim.utilities.parametrize import get_inertia_vals

    if TYPE_CHECKING:
        from bicycleparameters import Bicycle

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel

__all__ = ["RearFrameBase", "RigidRearFrame", "RigidRearFrameMoore"]


class RearFrameBase(ModelBase):
    """Base class for the rear frame of a bicycle."""

    @property
    @abstractmethod
    def steer_hub(self) -> Hub:
        """Steer hub expressed in the rear frame."""

    @property
    @abstractmethod
    def wheel_hub(self) -> Hub:
        """Wheel hub for the rear wheel expressed in the rear frame."""

    @property
    @abstractmethod
    def saddle(self) -> Attachment:
        """Attachment representing the saddle."""

    @property
    @abstractmethod
    def bottom_bracket(self) -> Point:
        """Point representing the center of the bottom bracket."""

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        ax_l = 0.15 * self.bottom_bracket.pos_from(self.wheel_hub.point).magnitude()
        saddle_low = self.saddle.point.locatenew(
            "P", 0.15 * self.bottom_bracket.pos_from(self.saddle.point))
        points = [
            self.bottom_bracket,
            saddle_low,
            self.wheel_hub.point.locatenew("P", -ax_l / 2 * self.wheel_hub.axis),
            self.bottom_bracket,
            self.wheel_hub.point.locatenew("P", ax_l / 2 * self.wheel_hub.axis),
            saddle_low,
            self.saddle.point,
            saddle_low,
            self.steer_hub.point.locatenew(  # not perfect but close enough
                "P", saddle_low.pos_from(self.steer_hub.point).dot(
                    self.steer_hub.axis) / 2 * self.steer_hub.axis),
            self.bottom_bracket,
        ]
        plot_object.add_line(points, self.name)
        for body in self.system.bodies:
            plot_object.add_body(body)


@set_default_convention("moore")
class RigidRearFrame(RearFrameBase):
    """Rigid rear frame."""

    @property
    def descriptions(self) -> dict[object, str]:
        """Dictionary of descriptions of the rear frame's symbols."""
        return {
            **super().descriptions,
            self.symbols["l_bbx"]: f"Distance between the rear hub and the bottom "
                                   f"bracket along {self.body.x}.",
            self.symbols["l_bbz"]: f"Distance between the rear hub and the bottom "
                                   f"bracket along {self.body.z}.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the rear frame."""
        super()._define_objects()
        self._body = RigidBody(self._add_prefix("body"))
        self.body.central_inertia = inertia(self.body.frame,
                                            *symbols(self._add_prefix("ixx iyy izz")),
                                            izx=Symbol(self._add_prefix("izx")))
        self._system = System.from_newtonian(self.body)
        self._saddle = Attachment(self.body.frame, Point(self._add_prefix("saddle")))
        self._bottom_bracket = Point(self._add_prefix("bottom_bracket"))
        self.symbols.update({name: Symbol(self._add_prefix(name))
                             for name in ("l_bbx", "l_bbz")})

    def _define_kinematics(self) -> None:
        """Define the kinematics of the rear frame."""
        super()._define_kinematics()
        self.bottom_bracket.set_pos(self.wheel_hub.point,
                                    self.symbols["l_bbx"] * self.body.x +
                                    self.symbols["l_bbz"] * self.body.z)
        self.bottom_bracket.set_vel(self.body.frame, 0)

    @property
    def body(self) -> RigidBody:
        """Rigid body representing the rear frame."""
        return self._body

    @property
    def saddle(self) -> Attachment:
        """Attachment representing the saddle."""
        return self._saddle

    @property
    def bottom_bracket(self) -> Point:
        """Point representing the center of the bottom bracket."""
        return self._bottom_bracket

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        if "Benchmark" in bicycle_parameters.parameters:
            bp = remove_uncertainties(bicycle_parameters.parameters["Benchmark"])
            params[self.body.mass] = bp["mB"]
        if "Measured" in bicycle_parameters.parameters:
            mep = remove_uncertainties(bicycle_parameters.parameters["Measured"])
            rr, lcs, hbb, lamht = (mep.get(name) for name in (
                "rR", "lcs", "hbb", "lamht"))
            if "mB" in mep:
                params[self.body.mass] = mep["mB"]
            if rr is None and "Benchmark" in bicycle_parameters.parameters:
                rr = bp["rR"]
            if lamht is None and "Benchmark" in bicycle_parameters.parameters:
                lamht = np.pi / 2 - bp["lam"]
            if not any(value is None for value in (rr, lcs, hbb, lamht)):
                glob_z = rr - hbb
                glob_x = np.sqrt(lcs ** 2 - glob_z ** 2)
                params[self.symbols["l_bbx"]] = (
                        glob_x * np.sin(lamht) - glob_z * np.cos(lamht))
                params[self.symbols["l_bbz"]] = (
                        glob_x * np.cos(lamht) + glob_z * np.sin(lamht))
        return params


class RigidRearFrameMoore(RigidRearFrame):
    """Rigid rear frame model based on Moore's convention."""

    convention: str = "moore"

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the attributes of the rear frame."""
        return {
            **super().descriptions,
            self.symbols["d1"]: "Perpendicular distance from the steer axis to the "
                                "center of the rear wheel (rear offset).",
            self.symbols["l1"]: "Distance in the rear frame x direction from the rear "
                                "wheel center to the center of mass of the rear frame.",
            self.symbols["l2"]: "Distance in the rear frame z direction from the rear "
                                "wheel center to the center of mass of the rear frame.",
            self.symbols["d4"]: "Distance in the rear frame x direction from the rear "
                                "wheel center to the point representing the saddle.",
            self.symbols["d5"]: "Distance in the rear frame z direction from the rear "
                                "wheel center to the point representing the saddle.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the rear frame."""
        super()._define_objects()
        self.symbols.update({
            name: Symbol(self._add_prefix(name))
            for name in ("d1", "l1", "l2", "d4", "d5")})
        self._steer_hub = Hub(
            self.body.frame, Point(self._add_prefix("steer_hub_point")), "z")
        self._wheel_hub = Hub(
            self.body.frame, Point(self._add_prefix("wheel_hub_point")), "y")

    def _define_kinematics(self) -> None:
        """Define the kinematics of the rear frame."""
        super()._define_kinematics()
        d1, l1, l2, d4, d5 = (self.symbols[name] for name in ("d1", "l1", "l2", "d4",
                                                              "d5"))
        x, z = self.body.x, self.body.z
        self.steer_hub.point.set_pos(self.wheel_hub.point, d1 * x)
        self.body.masscenter.set_pos(self.wheel_hub.point, l1 * x + l2 * z)
        self.saddle.point.set_pos(self.wheel_hub.point, d4 * x + d5 * z)
        self.body.masscenter.set_vel(self.body.frame, 0)
        self.steer_hub.point.set_vel(self.body.frame, 0)
        self.wheel_hub.point.set_vel(self.body.frame, 0)
        self.saddle.point.set_vel(self.body.frame, 0)

    @property
    def steer_hub(self) -> Hub:
        """Steer hub expressed in the rear frame."""
        return self._steer_hub

    @property
    def wheel_hub(self) -> Hub:
        """Wheel hub for the rear wheel expressed in the rear frame."""
        return self._wheel_hub

    def get_param_values(
        self,
        bicycle_parameters: Bicycle,
        *,
        include_rider: bool = False,
    ) -> dict[Symbol, float]:
        """Get the parameter values of the rear frame.

        Parameters
        ----------
        bicycle_parameters : Bicycle
            Bicycle parameters object containing the parameters.
        include_rider : bool, optional
            Whether to include the rider inertia parameters as part of the rear frame
            inertia, by default False. Hint: you can retrieve all paramters and specify
            this option using something like: ``bike.get_param_values(params) |
            bike.rear_frame.get_param_values(params, include_rider=True)``
        """
        params = super().get_param_values(bicycle_parameters)
        if "Benchmark" in bicycle_parameters.parameters:
            if (
                not include_rider
                and "Measured" in bicycle_parameters.parameters
            ):
                bp = remove_uncertainties(calculate_benchmark_from_measured(
                    bicycle_parameters.parameters["Measured"])[0])
            else:
                bp = remove_uncertainties(bicycle_parameters.parameters["Benchmark"])

            mop = benchmark_to_moore(bp)
            params[self.body.mass] = mop["mc"]
            params.update(get_inertia_vals(
                self.body, mop["ic11"], mop["ic22"], mop["ic33"], mop["ic12"],
                mop["ic23"], mop["ic31"]))
            params[self.symbols["d1"]] = mop["d1"]
            params[self.symbols["l1"]] = mop["l1"]
            params[self.symbols["l2"]] = mop["l2"]
        if "Measured" in bicycle_parameters.parameters:
            mep = remove_uncertainties(bicycle_parameters.parameters["Measured"])
            rr, lcs, hbb, lst, lsp, lamst, lamht = (mep.get(name) for name in (
                "rR", "lcs", "hbb", "lst", "lsp", "lamst", "lamht"))
            if rr is None and "Benchmark" in bicycle_parameters.parameters:
                rr = bp["rR"]
            if lamht is None and "Benchmark" in bicycle_parameters.parameters:
                lamht = np.pi / 2 - bp["lam"]
            if not any(value is None for value in (rr, lcs, hbb, lst, lsp, lamst)):
                r_rc_sdl = (yeadon_vec_to_bicycle_vec(
                    np.matrix([[0], [0], [0]]), mep, bp) + np.matrix([[0], [0], [rr]]))
                params[self.symbols["d4"]] = (
                        r_rc_sdl[0, 0] * np.sin(lamht) - r_rc_sdl[2, 0] * np.cos(lamht))
                params[self.symbols["d5"]] = (
                        r_rc_sdl[0, 0] * np.cos(lamht) + r_rc_sdl[2, 0] * np.sin(lamht))
        return params
