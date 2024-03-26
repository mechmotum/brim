"""Module containing the base class for all models in BRiM."""
from __future__ import annotations

from abc import ABCMeta
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from sympy import Basic, MutableDenseMatrix, Symbol, symbols
from sympy.physics.mechanics import System, dynamicsymbols, find_dynamicsymbols

from brim.core.auxiliary import AuxiliaryDataHandler
from brim.core.registry import Registry

try:  # pragma: no cover
    from bicycleparameters import Bicycle
except ImportError:  # pragma: no cover
    Bicycle = None
try:
    from symmeplot.matplotlib.plot_base import MplPlotBase
except ImportError:  # pragma: no cover
    MplPlotBase = None

if TYPE_CHECKING:
    from brim.core.requirement import ConnectionRequirement, ModelRequirement

__all__ = ["ConnectionBase", "ConnectionMeta", "LoadGroupBase", "LoadGroupMeta",
           "ModelBase", "ModelMeta", "set_default_convention"]


def _get_requirements(bases, namespace, req_attr_name):
    requirements = {}
    for base_cls in bases:
        base_reqs = getattr(base_cls, req_attr_name, None)
        if base_reqs is not None:
            for req in base_reqs:
                requirements[req.attribute_name] = req
    if req_attr_name in namespace:
        for req in namespace[req_attr_name]:
            requirements[req.attribute_name] = req
    return tuple(requirements.values())


def _create_submodel_property(requirement: ModelRequirement) -> property:
    def getter(self):
        return getattr(self, f"_{requirement.attribute_name}")

    def setter(self, model):
        if not (model is None or isinstance(model, requirement.types)):
            raise TypeError(
                f"{requirement.full_name} should be an instance of an subclass of "
                f"{requirement.type_name}, but {model!r} is an instance of "
                f"{type(model)}.")
        setattr(self, f"_{requirement.attribute_name}", model)

    getter.__annotations__ = {"return": requirement.type_hint}
    setter.__annotations__ = {"model": requirement.type_hint, "return": None}
    return property(getter, setter, None, requirement.description)


def _create_connection_property(requirement: ConnectionRequirement) -> property:
    def getter(self):
        return getattr(self, f"_{requirement.attribute_name}")

    def setter(self, conn):
        if not (conn is None or isinstance(conn, requirement.types)):
            raise TypeError(
                f"{requirement.full_name} should be an instance of an subclass "
                f"of {requirement.type_name}, but {conn!r} is an instance of "
                f"{type(conn)}.")
        setattr(self, f"_{requirement.attribute_name}", conn)

    getter.__annotations__ = {"return": requirement.type_hint}
    setter.__annotations__ = {"conn": requirement.type_hint, "return": None}
    return property(getter, setter, None, requirement.description)


class ModelMeta(ABCMeta):
    """Metaclass for the :class:`brim.core.model_base.ModelBase`."""

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: N804
        """Create a new class."""
        # Create properties for each of the requirements
        requirements = _get_requirements(bases, namespace, "required_models")
        for req in requirements:
            namespace[req.attribute_name] = _create_submodel_property(req)
        namespace["required_models"] = tuple(requirements)  # Update the requirements
        # Create properties for each of the requirements
        requirements = _get_requirements(bases, namespace, "required_connections")
        for req in requirements:
            namespace[req.attribute_name] = _create_connection_property(req)
        namespace["required_connections"] = tuple(requirements)  # Update
        instance = super().__new__(mcs, name, bases, namespace, **kwargs)
        Registry().register_model(instance)
        return instance


class ConnectionMeta(ABCMeta):
    """Metaclass for the :class:`brim.core.model_base.ConnectionBase`."""

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: N804
        """Create a new class."""
        # Create properties for each of the requirements
        requirements = _get_requirements(bases, namespace, "required_models")
        for req in requirements:
            namespace[req.attribute_name] = _create_submodel_property(req)
        namespace["required_models"] = tuple(requirements)  # Update the requirements
        instance = super().__new__(mcs, name, bases, namespace, **kwargs)
        Registry().register_connection(instance)
        return instance


class LoadGroupMeta(ABCMeta):
    """Metaclass for the :class:`brim.core.model_base.LoadGroupBase`."""

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: N804
        """Create a new class."""
        instance = super().__new__(mcs, name, bases, namespace, **kwargs)
        Registry().register_load_group(instance)
        return instance


class BrimBase:
    """Base class defining a common interface for the models and connections."""

    def __init__(self, name: str) -> None:
        """Create a new instance.

        Parameters
        ----------
        name : str
            Name of the object.
        """
        if not name.isidentifier():
            raise ValueError("The name of an object should be a valid identifier.")
        self._name = str(name)
        self._system = None
        self._auxiliary_handler = None
        self.symbols: dict[str, Any] = {}
        self.q: MutableDenseMatrix = MutableDenseMatrix()
        self.u: MutableDenseMatrix = MutableDenseMatrix()
        self.u_aux: MutableDenseMatrix = MutableDenseMatrix()

    def _add_prefix(self, names: str) -> str:
        """Add the name of the object as a prefix to a set of names.

        Explanation
        -----------
        Helper function to add the name of the object as a prefix to a set of names.
        This is used to create unique names for the objects in the model.
        """
        syms = symbols(names)
        if isinstance(syms, tuple):
            return ", ".join(f"{self.name}_{sym.name}" for sym in syms)
        return f"{self.name}_{syms.name}"

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"

    @property
    def descriptions(self) -> dict[Any, str]:
        """Descriptions of the attributes of the object."""
        return {}

    def get_description(self, obj: Any) -> str:
        """Get description of a given object."""
        if obj in self.descriptions:
            return self.descriptions[obj]
        if hasattr(self, "submodels"):
            for submodel in self.submodels:
                desc = submodel.get_description(obj)
                if desc is not None:
                    return desc
        if hasattr(self, "connections"):
            for conn in self.connections:
                desc = conn.get_description(obj)
                if desc is not None:
                    return desc
        if hasattr(self, "load_groups"):
            for load_group in self.load_groups:
                desc = load_group.get_description(obj)
                if desc is not None:
                    return desc

    def get_all_symbols(self) -> set[Basic]:
        """Get all declared symbols of a model."""
        syms = set()
        # Get local symbols.
        for sym in self.symbols.values():
            # Extract symbols if an expression is set as symbol.
            if isinstance(sym, Basic):
                syms.update(sym.free_symbols)
                syms.update(find_dynamicsymbols(sym))
        if dynamicsymbols._t in syms:  # Remove t.
            syms.remove(dynamicsymbols._t)
        # Traverse children.
        if hasattr(self, "submodels"):
            for submodel in self.submodels:
                syms.update(submodel.get_all_symbols())
        if hasattr(self, "connections"):
            for conn in self.connections:
                syms.update(conn.get_all_symbols())
        if hasattr(self, "load_groups"):
            for load_group in self.load_groups:
                syms.update(load_group.get_all_symbols())
        return syms

    @property
    def system(self) -> System | None:
        """System object used to store the information of the model itself.

        Notes
        -----
        This system object is used to store the information of the model itself. It does
        not by definition contain any information about the submodels or connections.
        Therefore, one cannot use this system object to form the equations of motion.
        Instead, one should use the :meth:`to_system` method to get the system object
        representing the entire model.
        """
        return self._system

    @property
    def auxiliary_handler(self) -> AuxiliaryDataHandler | None:
        """Auxiliary data handler of the model."""
        return self._auxiliary_handler

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        if Bicycle is None:
            raise ImportError("The bicycle parameters package is not installed.")
        return {}

    def set_plot_objects(self, plot_object: MplPlotBase) -> None:
        """Set the symmeplot plot objects."""
        if MplPlotBase is None:
            raise ImportError("The symmeplot package is not installed.")

    def _define_objects(self) -> None:
        """Define the objects of the system."""

    def define_objects(self) -> None:
        """Define the objects of the system."""
        self._define_objects()

    def _define_kinematics(self) -> None:
        """Define the kinematics of the system."""

    def define_kinematics(self) -> None:
        """Define the kinematics of the system."""
        self._define_kinematics()

    def _define_loads(self) -> None:
        """Define the loads of the system."""

    def define_loads(self) -> None:
        """Define the loads of the system."""
        self._define_loads()

    def _define_constraints(self) -> None:
        """Define the constraints of the system."""

    def define_constraints(self) -> None:
        """Define the constraints of the system."""
        self._define_constraints()


class ModelBase(BrimBase, metaclass=ModelMeta):
    """Base class for all objects in brim."""

    required_models: tuple[ModelRequirement, ...] = ()
    required_connections: tuple[ConnectionRequirement, ...] = ()
    convention: str = ""

    def __init__(self, name: str) -> None:
        """Create a new instance of the model.

        Parameters
        ----------
        name : str
            Name of the model.
        """
        super().__init__(name)
        self.is_root: bool | None = None  # None means that it is not defined.
        self._load_groups = []
        for req in self.required_models:
            setattr(self, f"_{req.attribute_name}", None)
        for req in self.required_connections:
            setattr(self, f"_{req.attribute_name}", None)

    @property
    def submodels(self) -> tuple[ModelBase]:
        """Submodels out of which this model exists."""
        submodels = []
        for req in self.required_models:
            submodels.append(getattr(self, req.attribute_name))
        return tuple(smd for smd in submodels if smd is not None)  # type: ignore

    @property
    def connections(self) -> tuple[ConnectionBase]:
        """Submodels out of which this model exists."""
        connections = []
        for req in self.required_connections:
            connections.append(getattr(self, req.attribute_name))
        return tuple(conn for conn in connections if conn is not None)  # type: ignore

    @property
    def load_groups(self) -> tuple[LoadGroupBase]:
        """Load groups of the connection."""
        return tuple(self._load_groups)  # type: ignore

    def add_load_groups(self, *load_groups: LoadGroupBase) -> None:
        """Add load groups to the connection."""
        for load_group in load_groups:
            load_group.parent = self
        self._load_groups.extend(load_groups)

    @classmethod
    def from_convention(cls, convention: str, name: str, *args, **kwargs) -> ModelBase:
        """Create a model from a convention."""
        possible_models = []
        for model in Registry().models:
            if issubclass(model, cls) and model.convention == convention:
                possible_models.append(model)
        if len(possible_models) == 0:
            raise ValueError(f"No model found for convention {convention!r} of type "
                             f"{cls}.")
        if len(possible_models) > 1:
            raise ValueError(f"Multiple models found for convention {convention!r} "
                             f"of type {cls}: {set(possible_models)}.")
        return possible_models[0](name, *args, **kwargs)

    def _set_auxiliary_handler(self, auxiliary_handler: AuxiliaryDataHandler) -> None:
        """Set the auxiliary data handler of the model."""
        self._auxiliary_handler = auxiliary_handler
        for submodel in self.submodels:
            submodel._set_auxiliary_handler(auxiliary_handler)
        for child in self.connections + self.load_groups:
            child._auxiliary_handler = auxiliary_handler

    def _define_connections(self) -> None:
        """Define the submodels used by connections in the model."""

    def define_connections(self) -> None:
        """Define the submodels used by connections in the model."""
        self._define_connections()
        for submodel in self.submodels:
            submodel.define_connections()

    def define_objects(self) -> None:
        """Initialize the objects belonging to the model."""
        if self.is_root is None:
            self.is_root = True
            queue = list(self.submodels)
            while queue:
                submodel = queue.pop(0)
                submodel.is_root = False
                queue.extend(submodel.submodels)
        for submodel in self.submodels:
            submodel.define_objects()
        self._define_objects()
        for load_group in self._load_groups:
            load_group.define_objects()
        if self.is_root:
            self._set_auxiliary_handler(AuxiliaryDataHandler.from_system(self.system))

    def define_kinematics(self) -> None:
        """Establish the kinematics of the objects belonging to the model."""
        for submodel in self.submodels:
            submodel.define_kinematics()
        self._define_kinematics()
        for load_group in self._load_groups:
            load_group.define_kinematics()
        if self.is_root:
            self.auxiliary_handler.apply_speeds()
            self.system.add_auxiliary_speeds(*self.auxiliary_handler.auxiliary_speeds)

    def define_loads(self) -> None:
        """Define the loads that are acting upon the model."""
        for submodel in self.submodels:
            submodel.define_loads()
        self._define_loads()
        for load_group in self._load_groups:
            load_group.define_loads()
        if self.is_root:
            self.system.add_loads(*self.auxiliary_handler.create_loads())

    def define_constraints(self) -> None:
        """Define the constraints on the model."""
        for submodel in self.submodels:
            submodel.define_constraints()
        self._define_constraints()
        for load_group in self._load_groups:
            load_group.define_constraints()

    def define_all(self) -> None:
        """Define all aspects of the model."""
        self.define_connections()
        self.define_objects()
        self.define_kinematics()
        self.define_loads()
        self.define_constraints()

    def get_param_values(self, bicycle_parameters: Bicycle) -> dict[Symbol, float]:
        """Get a parameters mapping of a model based on a bicycle parameters object."""
        params = super().get_param_values(bicycle_parameters)
        for submodel in self.submodels:
            params.update(submodel.get_param_values(bicycle_parameters))
        for conn in self.connections:
            params.update(conn.get_param_values(bicycle_parameters))
        return params

    def to_system(self) -> System:
        """Export the model to a single system instance."""

        def get_systems(model):
            """Get the systems of the submodels."""
            return ([model.system] + [conn.system for conn in model.connections] +
                    [s for submodel in model.submodels for s in get_systems(submodel)])

        return _merge_systems(*get_systems(self))


class ConnectionBase(BrimBase, metaclass=ConnectionMeta):
    """Base class for all connections in brim."""

    required_models: tuple[ModelRequirement, ...] = ()

    def __init__(self, name: str) -> None:
        """Create a new instance of the connection.

        Parameters
        ----------
        name : str
            Name of the connection.
        """
        super().__init__(name)
        self._load_groups = []
        for req in self.required_models:
            setattr(self, f"_{req.attribute_name}", None)

    @property
    def submodels(self) -> tuple[ModelBase]:
        """Submodels of the connection."""
        submodels = []
        for req in self.required_models:
            submodels.append(getattr(self, req.attribute_name))
        return tuple(smd for smd in submodels if smd is not None)  # type: ignore

    @property
    def load_groups(self) -> tuple[LoadGroupBase]:
        """Load groups of the connection."""
        return tuple(self._load_groups)  # type: ignore

    def add_load_groups(self, *load_groups: LoadGroupBase) -> None:
        """Add load groups to the connection."""
        for load_group in load_groups:
            load_group.parent = self
        self._load_groups.extend(load_groups)

    def define_objects(self) -> None:
        """Define the objects in the connection."""
        self._define_objects()
        for load_group in self._load_groups:
            load_group.define_objects()

    def define_kinematics(self) -> None:
        """Define the kinematics of the connection."""
        self._define_kinematics()
        for load_group in self._load_groups:
            load_group.define_kinematics()

    def define_loads(self) -> None:
        """Define the loads on the connection."""
        self._define_loads()
        for load_group in self._load_groups:
            load_group.define_loads()

    def define_constraints(self) -> None:
        """Define the constraints on the connection."""
        self._define_constraints()
        for load_group in self._load_groups:
            load_group.define_constraints()


class LoadGroupBase(BrimBase, metaclass=LoadGroupMeta):
    """Base class for the load groups."""

    required_parent_type: type | tuple[type, ...] = (ModelBase, ConnectionBase)

    def __init__(self, name: str) -> None:
        """Create a new instance of the load group.

        Parameters
        ----------
        name : str
            Name of the load group.
        """
        super().__init__(name)
        self._parent = None

    @property
    def parent(self) -> ModelBase | ConnectionBase | None:
        """Parent model or connection."""
        return self._parent

    @parent.setter
    def parent(self, parent: ModelBase | ConnectionBase) -> None:
        if self._parent is not None:
            raise ValueError(f"Load group is already used by {self.parent}")
        elif not isinstance(parent, self.required_parent_type):
            raise TypeError(
                f"Parent of {self} should be an instance of an subclass of "
                f"{self.required_parent_type}, but {parent!r} is an instance of "
                f"{type(parent)}.")
        self._parent = parent

    @property
    def system(self) -> System | None:
        """System object used to store the information of the model itself."""
        return self.parent.system if self.parent is not None else None


def set_default_convention(
        convention: str) -> Callable[[type[ModelBase]], type[ModelBase]]:
    """Set the default convention for a model."""

    def decorator(model: type[ModelBase]) -> type[ModelBase]:
        old_new = model.__new__

        @wraps(old_new)
        def new_new(cls, *args, **kwargs) -> ModelBase:
            if cls is model:
                return cls.from_convention(convention, *args, **kwargs)
            return old_new(cls)

        if not issubclass(model, ModelBase):
            raise TypeError(f"Model {model} is not a subclass of ModelBase.")
        model.__new__ = new_new
        return model

    return decorator


def _merge_systems(*systems: System) -> System:
    """Combine multiple system instance into one.

    Notes
    -----
    This function is not used in the current implementation of brim.
    However, it should in the end be moved to sympy mechanics.
    """
    system = System(systems[0].frame, systems[0].fixed_point)
    for s in systems:
        if s is None:  # pragma: no cover
            continue
        attributes = [
            ("q_ind", "q", "add_coordinates", {"independent": True}),
            ("q_dep", "q", "add_coordinates", {"independent": False}),
            ("u_ind", "u", "add_speeds", {"independent": True}),
            ("u_dep", "u", "add_speeds", {"independent": False}),
            ("u_aux", "u", "add_auxiliary_speeds", {}),
            ("bodies", "bodies", "add_bodies", {}),
            ("joints", "joints", "add_joints", {}),
            ("loads", "loads", "add_loads", {}),
            ("actuators", "actuators", "add_actuators", {}),
            ("kdes", "kdes", "add_kdes", {}),
            ("holonomic_constraints", "holonomic_constraints",
             "add_holonomic_constraints", {}),
            ("nonholonomic_constraints", "nonholonomic_constraints",
             "add_nonholonomic_constraints", {}),
        ]
        for attr_to_add, attr_existing, add_method, kwargs in attributes:
            for item in getattr(s, attr_to_add):
                if item not in getattr(system, attr_existing):
                    getattr(system, add_method)(item, **kwargs)
        system.velocity_constraints = (
            system.velocity_constraints[:] + s.velocity_constraints[:])
    return system
