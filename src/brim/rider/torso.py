"""Module containing torso models."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from sympy import Symbol
from sympy.physics.mechanics import Point

from brim.core import ModelBase, NewtonianBodyMixin

if TYPE_CHECKING:
    from sympy.physics.mechanics import ReferenceFrame

__all__ = ["TorsoBase", "SimpleRigidTorso"]


class TorsoBase(NewtonianBodyMixin, ModelBase):
    """Base class for the torso of a rider."""

    @property
    def left_shoulder_point(self) -> Point:
        """Location of the left shoulder.

        Explanation
        -----------
        The left shoulder point is defined as the point where the left shoulder joint
        is located. This point is used by connections to connect the left arm to the
        torso.
        """
        return self._left_shoulder_point

    @property
    @abstractmethod
    def left_shoulder_frame(self) -> ReferenceFrame:
        """The left shoulder frame.

        Explanation
        -----------
        The left shoulder frame is defined as the frame that is attached to the left
        shoulder point. This frame is used by connections to connect the left arm to
        the torso.
        """

    @property
    def right_shoulder_point(self) -> Point:
        """Location of the right shoulder.

        Explanation
        -----------
        The right shoulder point is defined as the point where the right shoulder joint
        is located. This point is used by connections to connect the right arm to the
        torso.
        """
        return self._right_shoulder_point

    @property
    @abstractmethod
    def right_shoulder_frame(self) -> ReferenceFrame:
        """The right shoulder frame.

        Explanation
        -----------
        The right shoulder frame is defined as the frame that is attached to the right
        shoulder point. This frame is used by connections to connect the right arm to
        the torso.
        """

    def define_objects(self) -> None:
        """Define the objects."""
        super().define_objects()
        self._left_shoulder_point = Point(self._add_prefix("LSP"))
        self._right_shoulder_point = Point(self._add_prefix("RSP"))


class SimpleRigidTorso(TorsoBase):
    """A simple rigid torso.

    Explanation
    -----------
    The simple rigid torso models the torso as being rigid. The shoulder joints are
    located a shoulder width apart at a height of the shoulder height above the center
    of mass of the torso.
    """

    @property
    def descriptions(self) -> dict[Any, str]:
        """Descriptions of the objects."""
        return {
            **super().descriptions,
            self.symbols["shoulder_width"]: "Distance between the left and right "
                                            "shoulder joints.",
            self.symbols["shoulder_height"]: "Distance between the shoulder joints and "
                                             "center of mass of the the torso.",
        }

    def define_objects(self) -> None:
        """Define the objects."""
        super().define_objects()
        self.symbols["shoulder_width"] = Symbol(self._add_prefix("shoulder_width"))
        self.symbols["shoulder_height"] = Symbol(self._add_prefix("shoulder_height"))

    def define_kinematics(self) -> None:
        """Define the kinematics."""
        super().define_kinematics()
        w, h = self.symbols["shoulder_width"], self.symbols["shoulder_height"]
        self.left_shoulder_point.set_pos(self.body.masscenter,
                                         -w / 2 * self.y - h * self.z)
        self.right_shoulder_point.set_pos(self.body.masscenter,
                                          w / 2 * self.y - h * self.z)

    @property
    def left_shoulder_frame(self) -> ReferenceFrame:
        """The left shoulder frame."""
        return self.body.frame

    @property
    def right_shoulder_frame(self) -> ReferenceFrame:
        """The right shoulder frame."""
        return self.body.frame