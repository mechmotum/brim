.. _component_implementation_guide:

=====================================
Guidelines on Implementing Components
=====================================

This document describes the guidelines on implementing components. It assumes
familiarity with the component structure used in SymBRiM :cite:`stienstra_2023_brim` and
the usage of SymPy mechanics and SymBRiM, as explained in the first two tutorials
:ref:`tutorials`.

SymBRiM Components
------------------

.. raw:: html

    <svg class="align-center" viewBox="0 0 114.1879 81.337509" width="550px">
        <use xlink:href="../_static/core_uml.svg#coreUML"></use>
    </svg>

The above figure shows the UML diagram of the core components of SymBRiM. The three core
components all inherit from :class:`symbrim.core.base_classes.BrimBase`, as they share a
similar define structure. The three core components are:

- :class:`Models<symbrim.core.base_classes.ModelBase>` - a system with its own
  respective system boundaries, which can be made up of other models.
- :class:`Connections<symbrim.core.base_classes.ConnectionBase>` - a utility of models
  in describing the interaction between submodels.
- :class:`Load groups<symbrim.core.base_classes.LoadGroupBase>` - a collection of forces
  and torques associated with a model or connection.

A large part of what is being shared between these components are the definition steps.
The definition steps are decoupled steps that are used to define the component. The
definition steps are:

0. **Define connections** (only in
   :class:`models<symbrim.core.base_classes.ModelBase>`): Associate the submodels with
   the connections, such that the connections know the submodels they will operate on.
1. **Define objects**: Create the objects, such as symbols reference frames, without
   defining any relationships between them.
2. **Define kinematics**: Establish relationships between the objects'
   orientations/positions, velocities, and accelerations.
3. **Define loads**: Specifies the forces and torques acting upon the system.
4. **Define constraints**: Computes the holonomic and nonholonomic constraints to which
   the system is subject.

The image below shows a schematic visualization of these steps for a rolling disc.

.. raw:: html

    <svg class="align-center" viewBox="0 0 180.37601 43.00629" width="800px">
        <use xlink:href="../_static/definition_steps.svg#definitionSteps"></use>
    </svg>

Usage of Base Classes
---------------------

SymBRiM uses base classes for components to specify a common structure of a component.
:class:`symbrim.bicycle.grounds.GroundBase` is an example of a base class. The advantage
of using base classes is that it allows for a common interface between components, which
makes it possible to swap out components without having to change the code. For example,
one can swap out the ground model for a different ground model without having to change
the code of the bicycle model.

In case of :class:`symbrim.bicycle.grounds.GroundBase` some of the commonly shared
properties are defined in the base class, such as a rigid body to represent the ground.
Apart from those it also prescribes several properties using :class:`abc.ABCMeta` and
:class:`abc.abstractmethod`. An example is the
:meth:`symbrim.bicycle.grounds.GroundBase.get_normal` method. These kind of abstract
methods have to be implemented by subclasses, such as
:class:`symbrim.bicycle.grounds.FlatGround`.

Setting Submodels and Connections
---------------------------------

To specify the submodels a model or connection requires, one should specify the class
property ``required_models``. This property should be a tuple of
:class:`symbrim.core.requirement.ModelRequirement`. Based on these requirements, the
metaclass automatically creates properties for each of the required submodels on
runtime. The following simple class shows how to specify the required submodels. ::

    class MyModel(ModelBase):
        """My model."""

        required_models: tuple[ModelRequirement, ...] = (
            ModelRequirement("ground", GroundBase, "Submodel of the ground."),
            ModelRequirement("other_submodel", OtherSubModel, "Other submodel."),
        )
        # These type hints are useful for some IDEs.
        ground: GroundBase
        other_submodel: OtherSubModel

The property created for ``"ground"`` will be like the following: ::

    @property
    def ground(self) -> GroundBase:
        """Submodel of the ground."""
        return self._ground

    @ground.setter
    def ground(self, model: GroundBase) -> None:
        """Submodel of the ground."""
        if not (model is None or isinstance(model, GroundBase)):
            raise TypeError(
                f"Ground should be an instance of an subclass of GroundBase, "
                f"but {model!r} is an instance of {type(model)}."
            )
        self._ground = model

Connections should be specified similarly with the class property
``required_connections``, using
:class:`symbrim.core.requirement.ConnectionRequirement`. ::

    class MyModel(ModelBase):
        """My model."""

        required_connections: tuple[ConnectionRequirement, ...] = (
            ConnectionRequirement("connection", MyConnection, "Connection."),
        )
        # These type hints are useful for some IDEs.
        connection: MyConnection

Specify a Load Group Parent
---------------------------

Load groups are associated with a certain model or connection. To specify this
association, one should specify the class property ``required_parent_type``. This
property is utilized in a ``isinstance(parent, self.required_parent_type)`` check when
adding a load group to a component. An example is shown below. ::

    class MyLoadGroup(LoadGroupBase):
        """My load group."""

        required_parent_type: type[Union[ModelBase, ConnectionBase]] = MyModel

    model = MyModel("my_model")
    load_group = MyLoadGroup("my_load_group")
    model.add_load_group(load_group)
    assert load_group.parent is model

    class MyModel2(ModelBase):
        """Some other model."""

    model2 = MyModel2("my_model2")
    load_group = MyLoadGroup("my_load_group")
    model2.add_load_group(load_group)  # Raises an error.

Implementation Define Steps
---------------------------

To implement the "define" steps in a model, connection, or load group, a leading
underscore is added to the method name. For example, ``_define_<step>``. These methods
solely implement the "define" step for the component itself without traversing the
submodels and load groups. The base classes, like
:class:`symbrim.core.base_classes.BrimBase`, contain the implementation of the "define"
methods, including traversal, which should be called by the user. These methods follow
the format ``define_<step>``.

We have established several helping guidelines for each of the define steps. The
subsections below discuss each of these per define step, and provide general coding
examples of the expected implementation.

Define Connections
~~~~~~~~~~~~~~~~~~

- If a connection is used, then the submodels of the connection are defined in the
  ``_define_connections`` method. ::

    def _define_connections(self) -> None:
        """Define the connections between the submodels."""
        super()._define_connections()
        self.connection.submodel = self.submodel

Define Objects
~~~~~~~~~~~~~~

- Each model and connection must instantiate its own
  :class:`sympy.physics.mechanics.system.System` instance and assign it to
  ``self._system``. Load groups automatically inherit the system from their parent.
- Symbols, such as masses and lengths, must be added to the ``self.symbols`` dictionary
  with a string as key and the (dynamic)symbol as value.
- Generalized coordinates must be set/added to the mutable ``self.q`` matrix.
- Generalized speeds must be set/added to the mutable ``self.u`` matrix.
- Auxiliary speeds must be set/added to the mutable ``self.uaux`` matrix.
- The name of each symbol, generalized coordinate, and generalized speed must be created
  using :meth:`symbrim.core.base_classes.BrimBase._add_prefix`. This method puts the
  name of the component in front of the symbol name, such that the symbol name is
  unique.
- Each symbol, generalized coordinate, and generalized speed must have a description in
  the :meth:`symbrim.core.base_classes.BrimBase.descriptions` property.
- The define objects step for each connection should be called manually because there
  could be dependencies between the define step of a connection and its parent model,
  it is utility after all.
- Other component specific objects, such as bodies and reference frames, must be defined
  in this stage, but they should not be oriented or positioned yet. ::

    @property
    def descriptions(self) -> dict[object, str]:
        """Descriptions of the attributes of the object."""
        return {
            **super().descriptions,
            self.symbols["symbol_name"]: "Description of the symbol.",
            self.symbols["f_noncontrib"]: "Description of the noncontributing force.",
            self.q[0]: f"First generalized coordinate of {self.name}.",
            self.q[1]: f"Second generalized coordinate of {self.name}.",
            self.u[0]: f"First generalized speed of {self.name}.",
            self.u[1]: f"Second generalized speed of {self.name}.",
            self.uaux[0]: f"Auxiliary speed of {self.name}.",
        }

    def _define_objects(self) -> None:
        """Define the objects of the system."""
        super()._define_objects()
        # Create symbols and generalized coordinates and speeds.
        self.symbols["symbol_name"] = symbols(self._add_prefix("symbol"))
        self.symbols["f_noncontrib"] = symbols(self._add_prefix("f_noncontrib"))
        self.q = MutableMatrix([dynamicsymbols(self._add_prefix("q1:3"))])
        self.u = MutableMatrix([dynamicsymbols(self._add_prefix("u1:3"))])
        self.uaux = MutableMatrix([dynamicsymbols(self._add_prefix("uaux"))])
        # Instantiate system.
        self._system = System()
        # Call define objects of connections.
        self.connection.define_objects()  # Without leading underscore!
        # Define other objects such as reference frames and bodies.
        ...

Define Kinematics
~~~~~~~~~~~~~~~~~

- It is generally best to first orient the reference frames in this step, and the
  position of the points. Next, one can optimize the definition of the velocities. With
  the introduction of the auxiliary data handler it is best practise to define the
  velocity of a point based on the point w.r.t. which it has been positioned.
  Parent models have to orient and define the submodels w.r.t. each other.
- The kinematical differential equations, generalized coordinates, and generalized
  speeds must be added to ``self.system``.
- Possibly one can also use joints for the above.
- Again the define kinematics step of each connection should be called manually.
- Generally, make sure to define the velocity of at least one point in the model's or
  connection's system.
- Noncontributing forces can be added to the auxiliary data handler. ::

    def _define_kinematics(self) -> None:
        """Define the kinematics of the system."""
        super()._define_kinematics()
        # Orient frames.
        self.frame.orient_axis(...)
        # Position points and set their velocities.
        self.point.set_pos(...)
        self.point.set_vel(self.system.frame, ...)
        # Add generalized coordinates, speeds, and kdes to the system.
        self.system.add_coordinates(*self.q)
        self.system.add_speed(*self.u)
        self.system.add_kdes(*(self.q.diff() - self.u))
        # Create and add joints.
        self.system.add_joints(...)
        # Call define kinematics of connections.
        self.connection.define_kinematics()  # Without leading underscore!
        # Add noncontributing force to the auxiliary data handler.
        self.auxiliary_data_handler.add_noncontributing_force(
          self.point, self.frame.x, self.uaux[0], self.symbols["f_noncontrib"])

Define Loads
~~~~~~~~~~~~

- As all points and reference frames have already been defined and positioned, this step
  only requires computation of the forces and torques and adding the to the
  ``self.system``.
- Noncontributing forces are fully handled by the auxiliary data handler.
- Again the define loads step of each connection should be called manually. ::

    def _define_loads(self) -> None:
        """Define the loads of the system."""
        super()._define_loads()
        # Add forces, torques and actuators
        self.system.add_loads(
            Force(self.point, ...),
            Torque(self.frame, ...),
            ...
        )
        self.system.add_actuators(...)
        # Call define loads of connections.
        self.connection.define_loads()  # Without leading underscore!

Define Constraints
~~~~~~~~~~~~~~~~~~

- For holonomic constraints a loop is being closed most of the time using a dot product.
  Just make sure to have some method to prevent the creation of constraint which are
  already satisfied. Optionally, you can use
  :class:`symbrim.utilities.utilities.check_zero`.
- For nonholonomic constraints the major difficulty is in the fact that one cannot
  assume anything about the already defined velocities. Especially points are
  susceptible to have multiple possible velocity definitions. Therefore, it is advised
  to compute the velocity based on the position graph of the points and the orientation
  and angular velocity graph of the reference frames. A good example of this is in the
  :class:`symbrim.bicycle.tires.NonHolonomicTire` class.
- To support usage of the object in a system with noncontributing forces it is also
  necessary to account for the auxiliary speeds. This can be done by specifically
  requesting the auxiliary velocity of a point and adding that to the constraint. Do
  also note that the ``velocity_constraints`` attribute is set to modify the velocity
  constraint resulting from the holonomic constraint. ::

    def _define_constraints(self) -> None:
        """Define the constraints of the system."""
        super()._define_constraints()
        self.system.add_holonomic_constraints(...)
        self.system.add_nonholonomic_constraints(...)
        # Overwrite the velocity constraints to include the auxiliary velocity.
        self.system.velocity_constraints = [
          self.system.holonomic_constraints[0].diff(dynamicsymbols._t) +
          self.auxiliary_handler.get_auxiliary_velocity(self.point).dot(...),
          *self.system.nonholonomic_constraints
        ]
        # Call define constraints of connections.
        self.connection.define_constraints()  # Without leading underscore!

Auxiliary Data Handler
----------------------

The :class:`symbrim.core.auxiliary.AuxiliaryDataHandler` is a utility class that is used
to compute noncontributing forces and optimize the computation of the velocity of
points. An instance of the auxiliary data handler is automatically created at the end of
the ``define_objects`` step. This instance is shared by the root model, i.e. the
uppermost parent model, with all submodels, connections, and load groups. This makes the
auxiliary data handler accessible from all components through the
``self.auxiliary_handler`` attribute.

In the ``define_kinematics`` step modelers can register noncontributing forces that
should be computed using the
:meth:`symbrim.core.auxiliary.AuxiliaryDataHandler.add_noncontributing_force` method.
This method requires the point, the axis of the force, the auxiliary speed, and the
force symbol as arguments. From this information the auxiliary data handler can do the
rest. When defining the other kinematics it is best practise to define the velocity of a
point based on the point w.r.t. which it has been positioned. This is because the
auxiliary data handler propagates the auxiliary velocities of points to other points
based on how points are defined w.r.t. to each other.

At the end of the ``define_kinematics`` step the auxiliary data handler automatically
computes the velocity of each point in the inertial frame, while adding the auxiliary
velocity. The auxiliary speed is also automatically added to the root model's system
instance.
At the end of the ``define_loads`` step the noncontributing forces are automatically
added to the root model's system instance.

When computing the constraints in the ``define_constraints`` step it is important to
take the auxiliary speeds into account, even if you didn't define any in you component.
In many cases it is possible that other components may have defined auxiliary speeds
that do affect your constraints. To get the auxiliary velocity of a point of intereset
you can use the
:meth:`symbrim.core.auxiliary.AuxiliaryDataHandler.get_auxiliary_velocity` method.
