"""Utility to compute the noncontributing forces and torques."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

from sympy import Function
from sympy.physics.mechanics import (
    Force,
    Point,
    ReferenceFrame,
    System,
    Torque,
    Vector,
    cross,
)

if TYPE_CHECKING:
    from sympy.physics.mechanics.loads import LoadBase

__all__ = ["AuxiliaryData", "AuxiliaryDataHandler"]


DynamicSymbol = Function


@dataclass(frozen=True)
class AuxiliaryData:
    """Dataclass to store noncontributing loads data.

    Parameters
    ----------
    location : Point | ReferenceFrame
        Location of the noncontributing load.
    direction : Vector
        Unit vector of the direction of the noncontributing load.
    speed_symbol : DynamicSymbol
        Auxiliary speed symbol used to compute of the noncontributing load.
    load_symbol : DynamicSymbol
        Magnitude of the noncontributing load.
    """

    location: Point | ReferenceFrame
    direction: Vector
    speed_symbol: DynamicSymbol
    load_symbol: DynamicSymbol

    def __post_init__(self) -> None:
        if not isinstance(self.location, (Point, ReferenceFrame)):
            raise TypeError(f"Location is of type {type(self.location)}, but must be an"
                            f" instance of {Point} or {ReferenceFrame}.")
        if not isinstance(self.direction, Vector):
            raise TypeError(
                f"The unit vector direction is of type {type(self.direction)}, but must"
                f" be an instance {Vector}.")
        if self.is_torque:
            raise NotImplementedError(
                "Noncontributing torques have not been implemented.")

    def get_load(self, inertial_frame: ReferenceFrame) -> Force | Torque:
        """Return the noncontributing load as a force or torque."""
        load_point = Point(f"{self.location.name}_aux")
        load_point.set_vel(inertial_frame, self.auxiliary_velocity)
        return Force(load_point, self.load_symbol * self.direction)

    @property
    def auxiliary_velocity(self) -> Vector:
        """Auxiliary velocity as vector."""
        return self.speed_symbol * self.direction

    @property
    def is_force(self) -> bool:
        """Boolean if the noncontributing load is a force."""
        return isinstance(self.location, Point)

    @property
    def is_torque(self) -> bool:
        """Boolean if the noncontributing load is a torque."""
        return isinstance(self.location, ReferenceFrame)


class AuxiliaryDataHandler:
    """Class to compute noncontributing loads in a system.

    Explanation
    -----------
    This is an experimental class to compute noncontributing forces and torques in a
    system. To do so, the class creates a tree representation of what points are used
    to compute the velocity of other points. Based on this tree, auxiliary velocities
    of each point are computed and added to the inertal velocity graph of the points.
    For this to work, the system must be defined in a way that the velocity of each
    point is computed based on the velocity of its parent point in the position tree.
    This is almost always the case. An example where this is not the case is when the
    the velocity of the front wheel of a bicycle, which is constrained to touch the
    ground using a holonomic constraint, is computed based on the assumption that its
    contact point is an instantaneous center of rotation.

    Parameters
    ----------
    inertial_frame : ReferenceFrame
        Inertial reference frame in which the equations of motion will be formed.
    inertial_point : Point
        Inertial point which is used as root in the tree graph representation.z
    """

    def __init__(self, inertial_frame: ReferenceFrame, inertial_point: Point):
        if not isinstance(inertial_frame, ReferenceFrame):
            raise TypeError("Inertial must be an instance of ReferenceFrame.")
        if not isinstance(inertial_point, Point):
            raise TypeError("Inertial point must be an instance of Point.")
        self._inertial_frame = inertial_frame
        self._inertial_point = inertial_point
        self._position_tree = None  # Chosen structure is {parent: [child1, ...]}
        self._aux_vels_points = None
        self.auxiliary_data_list: list[AuxiliaryData] = []

    @classmethod
    def from_system(cls, system: System) -> AuxiliaryDataHandler:
        """Create an auxiliary data handler from a system."""
        return cls(system.frame, system.fixed_point)

    @property
    def inertial_frame(self) -> ReferenceFrame:
        """Inertial reference frame."""
        return self._inertial_frame

    @property
    def inertial_point(self) -> Point:
        """Inertial point which is used as root in the tree graph representation."""
        return self._inertial_point

    @property
    def auxiliary_forces_data(self) -> tuple[AuxiliaryData]:
        """Tuple of noncontributing force data."""
        return tuple(ld for ld in self.auxiliary_data_list if ld.is_force)

    @property
    def auxiliary_torques_data(self) -> tuple[AuxiliaryData]:
        """Tuple of noncontributing torque data."""
        return tuple(ld for ld in self.auxiliary_data_list if ld.is_torque)

    @property
    def auxiliary_speeds(self) -> tuple[DynamicSymbol]:
        """Tuple of auxiliary speeds used to compute the noncontributing loads."""
        return tuple(ld.speed_symbol for ld in self.auxiliary_data_list)

    def add_noncontributing_force(self, point: Point, direction: Vector,
                                  speed_sym: DynamicSymbol, force_sym: DynamicSymbol
                                  ) -> AuxiliaryData:
        """Add the data of a noncontributing force to the graph."""
        force = AuxiliaryData(point, direction, speed_sym, force_sym)
        self.auxiliary_data_list.append(force)
        return force

    @staticmethod
    def _extract_tree(root: Any, get_childs: str | Callable[[Any], Iterable[Any]]
                      ) -> dict[Any: list[Any]]:
        """Create a tree graph using a breath-first search from the root."""
        if isinstance(get_childs, str):
            attr_name = get_childs
            def get_childs(parent: Any) -> Iterable[Any]:
                return getattr(parent, attr_name)
        tree = {}
        queue = [root]
        while queue:
            parent = queue.pop(0)
            if parent in tree:
                raise ValueError("Graph contains a cycle.")
            tree[parent] = []
            for neighbor in get_childs(parent):
                if neighbor not in tree:
                    queue.append(neighbor)
                    tree[parent].append(neighbor)
        return tree

    @staticmethod
    def _get_children_from_tree(tree: dict[Any, list[Any]], parent: Any,
                                include_parent: bool = False) -> list[Any]:
        """Get children of a node in a tree."""
        queue = tree[parent].copy()
        children = [parent] if include_parent else []
        while queue:
            child = queue.pop(0)
            children.append(child)
            queue.extend(tree[child])
        return children

    def retrieve_graphs(self) -> None:
        """Read in the graphs of the system."""
        self._position_tree = self._extract_tree(self.inertial_point, "_pos_dict")

    def _get_parent(self, point: Point) -> Point | None:
        """Get parent point in the position tree."""
        for parent, childs in self._position_tree.items():
            if point in childs:
                return parent

    def _compute_velocity(self, point: Point, parent: Point | None = None) -> Vector:
        """Compute the velocity of a point based on its parent in the position tree."""
        if self.inertial_frame in point._vel_dict:
            return point._vel_dict[self.inertial_frame]
        if point is self.inertial_point:
            return point.vel(self.inertial_frame)
        if self._position_tree is None or point not in self._position_tree:
            self.retrieve_graphs()
        if parent is None:
            parent = self._get_parent(point)
            if parent is None:
                raise ValueError(
                    f"Point {point!r} is not connected to {self.inertial_point!r}.")
        if self.inertial_frame not in parent._vel_dict:
            self._compute_velocity(parent)
        shared_frames = set(point._vel_dict).intersection(parent._vel_dict)
        # Compute velocity based on the velocity two point theorem if possible.
        for frame in shared_frames:
            if point._vel_dict[frame] == 0 and parent._vel_dict[frame] == 0:
                point.set_vel(self.inertial_frame,
                              self._compute_velocity(parent)
                              - cross(point.pos_from(parent),
                                      frame.ang_vel_in(self.inertial_frame)))
                return point._vel_dict[self.inertial_frame]
        # Compute velocity based on the velocity one point theorem if possible.
        for frame in shared_frames:
            if parent._vel_dict[frame] == 0 and frame in point._vel_dict:
                point.set_vel(self.inertial_frame,
                              self._compute_velocity(parent) + point._vel_dict[frame]
                              - cross(point.pos_from(parent),
                                      frame.ang_vel_in(self.inertial_frame)))
                return point._vel_dict[self.inertial_frame]
        # Fall back to velocity computation based on vector differentiation.
        point.set_vel(self.inertial_frame,
                      parent._vel_dict[self.inertial_frame] +
                      point.pos_from(parent).dt(self.inertial_frame))
        return point._vel_dict[self.inertial_frame]

    def apply_speeds(self) -> None:
        """Apply auxiliary speeds to the velocity graph."""
        if self._aux_vels_points is not None:
            raise ValueError("Auxiliary speeds have already been applied.")
        self.retrieve_graphs()
        all_points = self._get_children_from_tree(
            self._position_tree, self.inertial_point, include_parent=True)
        self._aux_vels_points = {pt: Vector(0) for pt in all_points}

        if self.auxiliary_torques_data:  # pragma: no cover
            raise NotImplementedError(
                "Support for noncontributing torques has not been implemented")

        # Add auxiliary velocities to auxiliary speed graph of the points
        for load in self.auxiliary_forces_data:
            if load.location not in self._position_tree:
                raise ValueError(
                    f"The point of the noncontributing force {load!r} is not connected"
                    f" to {self.inertial_point!r}.")
            for point in self._get_children_from_tree(
                self._position_tree, load.location, include_parent=True):
                self._aux_vels_points[point] += load.auxiliary_velocity

        # Set all speeds using a breath first search.
        # This is done before adding the auxiliary forces because auxiliary speeds may
        # otherwise also be added by Point.vel.
        queue = [self.inertial_point]
        while queue:
            parent = queue.pop(0)
            for child in self._position_tree[parent]:
                self._compute_velocity(child, parent)
                queue.append(child)

        # Add auxiliary speeds to each point of the graph.
        for point in all_points:
            aux_vel = self._aux_vels_points[point]
            if aux_vel != 0:
                point.set_vel(self.inertial_frame,
                              point._vel_dict[self.inertial_frame] + aux_vel)

    def get_auxiliary_velocity(self, point: Point) -> Vector:
        """Return the auxiliary velocity of a point."""
        if self._aux_vels_points is None:
            raise ValueError("Auxiliary velocities have not been computed yet.")
        elif point not in self._aux_vels_points:
            raise ValueError(
                f"Auxiliary velocity of point {point!r} has not been computed.")
        return self._aux_vels_points[point]

    def create_loads(self) -> list[LoadBase]:
        """Create loads for all noncontributing load data."""
        return [ld.get_load(self.inertial_frame) for ld in self.auxiliary_data_list]
