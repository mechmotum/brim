"""Module containing the models of the ground."""
from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING

from sympy.physics.mechanics import (
    Point,
    ReferenceFrame,
    RigidBody,
    System,
    Vector,
    cross,
)

from symbrim.core import ModelBase

if TYPE_CHECKING:
    import contextlib

    from sympy import Expr

    T_position = Point | Vector | tuple[Expr, ...]

    with contextlib.suppress(ImportError):
        from symbrim.utilities.plotting import PlotModel

__all__ = ["GroundBase", "FlatGround"]


class GroundBase(ModelBase):
    """Base class for the ground."""

    def _parse_plane_position(self, position: T_position) -> tuple[Expr, Expr, Expr]:
        """Parse the position of a point on the ground.

        Explanation
        -----------
        This is a utility to be used by subclasses to parse the position of a point
        passed to get_normal and get_tangent_vectors.

        Parameters
        ----------
        position : Point, Vector, tuple
            The position of the point on the ground.

        Returns
        -------
        tuple
            The position of the point on the ground expressed in the ground frame.
        """
        if isinstance(position, Point):
            position = position.pos_from(self.origin)
        if isinstance(position, Vector):
            position = position.to_matrix(self.frame)[:]
        position = tuple(position)
        if len(position) == 2:
            position = (*position, 0)
        if len(position) != 3:
            raise ValueError("Position must be a 2D or 3D vector.")
        return position

    def _define_objects(self) -> None:
        """Define the objects of the ground."""
        super()._define_objects()
        self._system = System(
            ReferenceFrame(self._add_prefix("frame")),
            Point(self._add_prefix("origin")),
        )
        self._body = RigidBody(self.name, self.origin, self.frame)

    def _define_kinematics(self) -> None:
        """Define the kinematics of the ground."""
        super()._define_kinematics()
        self.origin.set_vel(self.frame, 0)

    @property
    def body(self) -> RigidBody:
        """The body representing the ground."""
        warnings.warn(
            "The 'body' property is deprecated and will be removed in a future version."
            " Use 'system.frame' and 'system.fixed_point' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._body

    @property
    def frame(self) -> ReferenceFrame:
        """Frame fixed to the ground."""
        return self.system.frame

    @property
    def origin(self) -> Point:
        """Origin of the ground."""
        return self.system.fixed_point

    @abstractmethod
    def get_normal(self, position: T_position) -> Vector:
        """Get normal vector of the ground."""

    @abstractmethod
    def get_tangent_vectors(self, position: T_position) -> tuple[Vector, Vector]:
        """Get tangent vectors of the ground plane."""

    @abstractmethod
    def set_pos_point(self, point: Point, position: T_position) -> None:
        """Set the location of a point on the ground."""

    def set_plot_objects(self, plot_object: PlotModel) -> None:
        """Set the symmeplot plot objects."""
        super().set_plot_objects(plot_object)
        plot_object.add_frame(self.frame, self.origin)


class FlatGround(GroundBase):
    """Flat ground."""

    def __init__(self, name: str, normal: str = "-z") -> None:
        """Initialize the flat ground.

        Parameters
        ----------
        name : str
            Name of the ground.
        normal : str, optional
            Normal vector of the ground. Options are:
            - "+x", "-x", "+y", "-y", "+z", "-z"
            The default is "-z".

        """
        super().__init__(name)
        self._normal = normal

    def _define_objects(self) -> None:
        """Define the objects of the ground."""
        super()._define_objects()
        if self._normal[0] == "-":
            self._normal = self._normal[1:]
            times = -1
        else:
            times = 1
            if self._normal[0] == "+":
                self._normal = self._normal[1:]
        self._normal = times * self.frame[self._normal]
        if cross(self._normal, self.frame.x) == 0:
            self._planar_vectors = (self.frame.y, self.frame.z)
        elif cross(self._normal, self.frame.y) == 0:
            self._planar_vectors = (self.frame.x, self.frame.z)
        else:
            self._planar_vectors = (self.frame.x, self.frame.y)

    def get_normal(self, position: T_position) -> Vector:  # noqa: ARG002
        """Get normal vector of the ground."""
        return self._normal

    def get_tangent_vectors(self, position: T_position) -> tuple[Vector, Vector]:  # noqa: ARG002
        """Get tangent vectors of the ground plane."""
        return self._planar_vectors

    def set_pos_point(self, point: Point, position: T_position) -> None:
        """Set the location of a point on the ground."""
        position = self._parse_plane_position(position)
        point.set_pos(self.origin, position[0] * self._planar_vectors[0] +
                      position[1] * self._planar_vectors[1])
