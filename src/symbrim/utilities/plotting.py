"""Module containing utilities to easily plot the model in SymMePlot.

Notes
-----
The plotter is based on SymMePlot, which is an experimental plotting library for SymPy
mechanics. Therefore, there is no guarantee that the plotter will work in the future.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from symmeplot.matplotlib import Scene3D
from symmeplot.matplotlib.plot_base import MplPlotBase
from typing_extensions import Self

from symbrim.core import ConnectionBase, LoadGroupBase, ModelBase

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from sympy import Expr
    from sympy.physics.mechanics import Point, ReferenceFrame

    from symbrim.core.base_classes import BrimBase

__all__ = ["Plotter", "PlotBrimMixin", "PlotModel", "PlotConnection", "PlotLoadGroup"]


class Plotter(Scene3D):
    """Plotter for models created by SymBRiM using SymMePlot."""

    @classmethod
    def from_model(cls, model: ModelBase, ax: Axes3D | None = None,
                   **inertial_frame_properties: dict[str, object]) -> Self:
        """Initialize the plotter.

        Parameters
        ----------
        model : ModelBase
            Model to plot.
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
            Axes to plot on. If None, a new figure is created.
        **inertial_frame_properties
            Keyword arguments are parsed to
            :class:`symmeplot.matplotlib.plot_objects.PlotFrame` representing the inertial
            reference frame.
        """
        plotter = cls(model.system.frame, model.system.fixed_point, ax=ax,
                      **inertial_frame_properties)
        plotter.add_model(model)
        return plotter

    def add_model(self, model: ModelBase, **kwargs: dict[str, object]) -> PlotModel:
        """Add a model to the plotter.

        Parameters
        ----------
        model : ModelBase
            Model to add.
        **kwargs
            Keyword arguments are passed to
            :class:`symbrim.utilities.plotting.PlotModel`.
        """
        obj = PlotModel(self.inertial_frame, self.zero_point, model, **kwargs)
        self.add_plot_object(obj)
        return obj

    def add_connection(
        self, connection: ConnectionBase, **kwargs: dict[str, object]
    ) -> PlotConnection:
        """Add a connection to the plotter.

        Parameters
        ----------
        connection : ConnectionBase
            Connection to add.
        **kwargs
            Keyword arguments are passed to
            :class:`symbrim.utilities.plotting.PlotConnection`.
        """
        obj = PlotConnection(self.inertial_frame, self.zero_point, connection, **kwargs)
        self.add_plot_object(obj)
        return obj

    def add_load_group(
        self, load_group: LoadGroupBase, **kwargs: dict[str, object]
    ) -> PlotLoadGroup:
        """Add a load group to the plotter.

        Parameters
        ----------
        load_group : LoadGroupBase
            Load group to add.
        **kwargs :
            Keyword arguments are passed to
            :class:`symbrim.utilities.plotting.PlotLoadGroup`.
        """
        obj = PlotLoadGroup(self.inertial_frame, self.zero_point, load_group, **kwargs)
        self.add_plot_object(obj)
        return obj


class PlotBrimMixin:
    """Mixin class for plotting SymBRiM objects."""

    _PlotPoint: type[MplPlotBase] = Scene3D._PlotPoint
    _PlotLine: type[MplPlotBase] = Scene3D._PlotLine
    _PlotVector: type[MplPlotBase] = Scene3D._PlotVector
    _PlotFrame: type[MplPlotBase] = Scene3D._PlotFrame
    _PlotBody: type[MplPlotBase] = Scene3D._PlotBody

    add_point = Plotter.add_point
    add_line = Plotter.add_line
    add_vector = Plotter.add_vector
    add_frame = Plotter.add_frame
    add_body = Plotter.add_body
    plot_objects = Plotter.plot_objects
    add_plot_object = Plotter.add_plot_object
    get_plot_object = Plotter.get_plot_object

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 brim_object: BrimBase) -> None:
        """Initialize a plot object of the SymBRiM model."""
        super().__init__(inertial_frame, zero_point, brim_object, brim_object.name)
        brim_object.set_plot_objects(self)

    def get_sympy_object_exprs(self) -> tuple[Expr, Expr, Expr]:
        """Get coordinate of the point as expressions."""
        if self.sympy_object.system is None:
            annot_point = self.zero_point
        else:
            annot_point = self.sympy_object.system.fixed_point
        return tuple(
            annot_point.pos_from(self.zero_point).to_matrix(self.inertial_frame)[:]
        )

    @property
    def annot_coords(self) -> np.ndarray[np.float64]:
        """Coordinate where the annotation text is displayed."""
        if not self._values:
            return np.zeros(3)
        return np.array(self._values[0]).reshape(3)


class PlotModel(PlotBrimMixin, MplPlotBase):
    """Plot object of a SymBRiM model."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 model: ModelBase, plot_load_groups: bool = True) -> None:
        """Initialize a plot object of a model.

        Parameters
        ----------
        inertial_frame : ReferenceFrame
            The reference frame with respect to which all objects will be oriented.
        zero_point : Point
            The absolute origin with respect to which all objects will be positioned.
        model : ModelBase
            The SymBRiM model, which is being plotted.
        plot_load_groups : bool, optional
            Whether to plot the load groups, by default True.
        """
        super().__init__(inertial_frame, zero_point, model)
        self.model = model
        for submodel in self.model.submodels:
            self._children.append(PlotModel(
                inertial_frame, zero_point, submodel, plot_load_groups))
        for connection in self.model.connections:
            self._children.append(PlotConnection(
                inertial_frame, zero_point, connection, False, plot_load_groups
            ))
        if plot_load_groups:
            for load_group in self.model.load_groups:
                self._children.append(PlotLoadGroup(
                    inertial_frame, zero_point, load_group))

    @property
    def model(self) -> ModelBase:
        """The SymBRiM model, which is being plotted."""
        return self._model

    @model.setter
    def model(self, model: ModelBase) -> None:
        if not isinstance(model, ModelBase):
            raise TypeError(f"'model' should be an instance of {ModelBase}.")
        self._model = model
        self._values = []

    @property
    def submodels(self) -> tuple[PlotModel, ...]:
        """Plot objects of the submodels."""
        return tuple(child for child in self._children if isinstance(child, PlotModel))

    @property
    def connections(self) -> tuple[PlotConnection, ...]:
        """Plot objects of the connections."""
        return tuple(child for child in self._children
                     if isinstance(child, PlotConnection))

    @property
    def load_groups(self) -> tuple[PlotLoadGroup, ...]:
        """Plot objects of the load groups."""
        return tuple(child for child in self._children
                     if isinstance(child, PlotLoadGroup))


class PlotConnection(PlotBrimMixin, MplPlotBase):
    """Plot object of a SymBRiM connection."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 connection: ConnectionBase, plot_submodels: bool = True,
                 plot_load_groups: bool = True) -> None:
        """Initialize a plot object of a connection.

        Parameters
        ----------
        inertial_frame : ReferenceFrame
            The reference frame with respect to which all objects will be oriented.
        zero_point : Point
            The absolute origin with respect to which all objects will be positioned.
        connection : ConnectionBase
            The SymBRiM connection, which is being plotted.
        plot_submodels : bool, optional
            Whether to plot the submodels, by default True.
        plot_load_groups : bool, optional
            Whether to plot the load groups, by default True.
        """
        super().__init__(inertial_frame, zero_point, connection)
        self.connection = connection
        if plot_submodels:
            for submodel in self.connection.submodels:
                self._children.append(PlotModel(
                    inertial_frame, zero_point, submodel, plot_load_groups))
        if plot_load_groups:
            for load_group in self.connection.load_groups:
                self._children.append(PlotLoadGroup(
                    inertial_frame, zero_point, load_group))

    @property
    def connection(self) -> ConnectionBase:
        """The SymBRiM connection, which is being plotted."""
        return self._connection

    @connection.setter
    def connection(self, connection: ConnectionBase) -> None:
        if not isinstance(connection, ConnectionBase):
            raise TypeError(f"'connection' should be an instance of {ConnectionBase}.")
        self._connection = connection
        self._values = []

    @property
    def submodels(self) -> tuple[PlotModel, ...]:
        """Plot objects of the submodels."""
        return tuple(child for child in self._children if isinstance(child, PlotModel))

    @property
    def load_groups(self) -> tuple[PlotLoadGroup, ...]:
        """Plot objects of the load groups."""
        return tuple(child for child in self._children
                     if isinstance(child, PlotLoadGroup))


class PlotLoadGroup(PlotBrimMixin, MplPlotBase):
    """Plot object of a SymBRiM load group."""

    def __init__(self, inertial_frame: ReferenceFrame, zero_point: Point,
                 load_group: LoadGroupBase) -> None:
        """Initialize a plot object of a load group.

        Parameters
        ----------
        inertial_frame : ReferenceFrame
            The reference frame with respect to which all objects will be oriented.
        zero_point : Point
            The absolute origin with respect to which all objects will be positioned.
        load_group : LoadGroupBase
            The SymBRiM load group, which is being plotted.
        """
        super().__init__(inertial_frame, zero_point, load_group)
        self.load_group = load_group

    @property
    def load_group(self) -> LoadGroupBase:
        """The SymBRiM load group, which is being plotted."""
        return self._load_group

    @load_group.setter
    def load_group(self, load_group: LoadGroupBase) -> None:
        if not isinstance(load_group, LoadGroupBase):
            raise TypeError(f"'load_group' should be an instance of {LoadGroupBase}.")
        self._load_group = load_group
        self._values = []
