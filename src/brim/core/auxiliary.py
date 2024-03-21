"""Utility to compute the noncontributing forces and torques."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

from sympy import Function
from sympy.physics.mechanics import Force, Point, ReferenceFrame, Torque, Vector

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

    Notes
    -----
    When defining the direction be careful that the reference part is the child, when
    visualizing the position or orientation graph as a tree. For example, if you have a
    four-bar linkage and you'd like to know the noncontributing force in the first joint
    then you can specify the direction as ``A1.x``. However, if you would like the
    noncontributing force in the closing link and you use the end effector as location
    then you should specify the direction to be negative ``-A1.x``.

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
    """Class to compute noncontributing loads in a system."""

    def __init__(self, inertial_frame: ReferenceFrame, inertial_point: Point):
        if not isinstance(inertial_frame, ReferenceFrame):
            raise TypeError("Inertial must be an instance of ReferenceFrame.")
        if not isinstance(inertial_point, Point):
            raise TypeError("Inertial point must be an instance of Point.")
        self._inertial_frame = inertial_frame
        self._inertial_point = inertial_point
        self._position_tree = None
        self._aux_vels_points = None
        self.auxiliary_data_list: list[AuxiliaryData] = []

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

    def add_noncontributing_force(self, point: Point, direction: Vector,
                                  speed_sym: DynamicSymbol, force_sym: DynamicSymbol
                                  ) -> AuxiliaryData:
        """Add a noncontributing force to the graph."""
        force = AuxiliaryData(point, direction, speed_sym, force_sym)
        self.auxiliary_data_list.append(force)
        return force

    @staticmethod
    def _extract_tree(root: Any, get_childs: str | Callable[[Any], Iterable[Any]],
                      _validate_tree: bool = True) -> dict[Any: list[Any]]:
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
        if _validate_tree:
            all_nodes = [root] + [obj for obj_lst in tree.values() for obj in obj_lst]
            if len(all_nodes) != len(set(all_nodes)):
                raise ValueError("Graph contains a cycle.")
            elif len(all_nodes) != len(tree):
                raise ValueError("Graph is invalid.")
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
        shared_frames = set(point._vel_dict).intersection(parent._vel_dict)
        # Compute velocity based on the velocity two point theorem if possible.
        for frame in shared_frames:
            if point._vel_dict[frame] == 0 and parent._vel_dict[frame] == 0:
                return point.v2pt_theory(parent, self.inertial_frame, frame)
        # Compute velocity based on the velocity one point theorem if possible.
        for frame in shared_frames:
            if parent._vel_dict[frame] == 0:
                return point.v1pt_theory(parent, self.inertial_frame, frame)
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

        if self.auxiliary_torques_data:
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
        parent = None
        while queue:
            point = queue.pop(0)
            self._compute_velocity(point, parent)
            queue.extend(self._position_tree[point])
            parent = point

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
        """Create loads for all noncontributing loads data."""
        return [ld.get_load(self.inertial_frame) for ld in self.auxiliary_data_list]