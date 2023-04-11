"""Module containing the base class for all models in BRiM."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from sympy import symbols

if TYPE_CHECKING:
    from sympy.physics.mechanics import System

    from brim.core.requirement import Requirement


class ModelMeta(ABCMeta):
    """Metaclass for the :class:`brim.core.model_base.ModelBase`."""

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: N804
        """Create a new class."""

        def create_submodel_property(requirement: Requirement) -> property:
            def getter(self):
                return getattr(self, f"_{requirement.attribute_name}")

            def setter(self, model):
                if not (model is None or isinstance(model, requirement.types)):
                    raise TypeError(
                        f"{requirement.full_name} should be an instance of an subclass "
                        f"of {requirement.type_name}, but {model} is an instance of "
                        f"{type(model)}.")
                setattr(self, f"_{requirement.attribute_name}", model)

            getter.__annotations__ = {"return": requirement.type_hint}
            setter.__annotations__ = {"model": requirement.type_hint, "return": None}
            return property(getter, setter, None, requirement.description)

        # Create properties for each of the requirements
        requirements = {}
        for base_cls in bases:
            base_reqs = getattr(base_cls, "requirements", None)
            if base_reqs is not None:
                for req in base_reqs:
                    requirements[req.attribute_name] = req
        if "requirements" in namespace:
            for req in namespace["requirements"]:
                requirements[req.attribute_name] = req
        requirements = list(requirements.values())
        for req in requirements:
            namespace[req.attribute_name] = create_submodel_property(req)
        namespace["requirements"] = tuple(requirements)  # update the requirements
        return super().__new__(mcs, name, bases, namespace, **kwargs)

    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class.

        Notes
        -----
        The formulation is removed from the keyword arguments before the instance is
        created. This is done to prevent the formulation from being passed to the
        ``__init__`` method.

        """
        obj = cls.__new__(cls, *args, **kwargs)
        if "formulation" in kwargs:
            del kwargs["formulation"]
        obj.__init__(*args, **kwargs)
        return obj


class ModelBase(metaclass=ModelMeta):
    """Base class for all objects in brim."""

    requirements: tuple[Requirement, ...] = ()

    def __init__(self, name: str):
        """Create a new instance of the model.

        Parameters
        ----------
        name : str
            Name of the model.

        """
        if not name.isidentifier():
            raise ValueError("The name of an object should be a valid identifier.")
        if not name:
            raise ValueError("The name of an object may not be empty.")
        self._name = str(name)
        for req in self.requirements:
            setattr(self, f"_{req.attribute_name}", None)
        self._system = None
        self.symbols: dict[str, Any] = {}
        self.define_objects()

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        """Name of the part of the model."""
        return self._name

    def _add_prefix(self, names: str) -> str:
        """Add the name of the model as a prefix to a set of names.

        Explanation
        -----------
        Helper function to add the name of the model as a prefix to a set of names. This
        is used to create unique names for the objects in the model.

        """
        syms = symbols(names)
        if isinstance(syms, tuple):
            return ", ".join(f"{self.name}_{sym.name}" for sym in syms)
        return f"{self.name}_{syms.name}"

    @property
    def submodels(self) -> frozenset[ModelBase]:
        """Submodels out of which this model exists."""
        submodels = []
        for req in self.requirements:
            submodels.append(getattr(self, req.attribute_name))
        return frozenset(submodel for submodel in submodels if submodel is not None)

    def get_description(self, obj: Any) -> str:
        """Get description of a given object."""
        if obj in self.descriptions:
            return self.descriptions[obj]
        for submodel in self.submodels:
            desc = submodel.get_description(obj)
            if desc is not None:
                return desc

    def add_mixin(self, mixin: type) -> None:
        """Extend model with a mixin class."""
        if not isinstance(mixin, type):
            raise TypeError("Mixin should be a class.")
        self.__class__ = type(self.__class__.__name__, (mixin, self.__class__), {})
        if hasattr(mixin, "requirements"):
            for req in mixin.requirements:
                setattr(self, f"_{req.attribute_name}", None)
        if hasattr(mixin, "define_objects"):
            mixin.define_objects(self)

    @property
    def descriptions(self) -> dict[Any, str]:
        """Descriptions of the attributes of the model."""
        return {}

    @property
    def system(self) -> System:
        """System object representing the model."""
        return self._system

    @abstractmethod
    def define_objects(self) -> None:
        """Initialize the objects belonging to the model."""
        for submodel in self.submodels:
            submodel.define_objects()

    @abstractmethod
    def define_kinematics(self) -> None:
        """Establish the kinematics of the objects belonging to the model."""
        for submodel in self.submodels:
            submodel.define_kinematics()

    @abstractmethod
    def define_loads(self) -> None:
        """Define the loads that are part of the model."""
        for submodel in self.submodels:
            submodel.define_loads()
