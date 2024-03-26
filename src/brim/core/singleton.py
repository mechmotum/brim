"""Singleton class."""
from __future__ import annotations

from typing import ClassVar

__all__ = ["SingletonMeta", "Singleton"]


class SingletonMeta(type):
    """Metaclass for Singleton."""

    _instances: ClassVar[dict[type, object]] = {}

    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def activate(cls, instance: object) -> None:
        """Activate the instance of the class."""
        cls._instances[cls] = instance

    def deactivate(cls, instance: object):
        """Deactivate the instance of the class."""
        if cls in cls._instances:  # pragma: no cover
            if cls._instances[cls] is instance:
                cls._instances.pop(cls)
            else:
                raise ValueError(
                    f"The instance {instance!r} is not the active instance of {cls!r}")


class Singleton(metaclass=SingletonMeta):
    """Makes a class a singleton via inheritance."""

    def activate(self) -> None:
        """Activate the instance of the class."""
        SingletonMeta.activate(type(self), self)

    def deactivate(self) -> None:
        """Deactivate the instance of the class."""
        SingletonMeta.deactivate(type(self), self)
