.. _introduction_to_brim:

=======================
Introduction to SymBRiM
=======================
..
    Paragraph: Quick introduction on what SymBRiM is and does.

SymBRiM :cite:`stienstra_2023_brim` is a Python package to model bicycle-riders
symbolically. It offers a modular and extendable framework that empowers users to easily
compose a custom bicycle-rider model. These models are build from a library of
components, such as knife-edge wheels or toroidal-shaped wheels. The components serve as
the building blocks of the system. Leveraging the open-source Python package
:mod:`sympy` :cite:`Meurer2017`, a computer algebra system, SymBRiM allows the usage of
:mod:`sympy`'s bodies and joints interface to modify the model and other tools to
manipulate the equations. The equations of motion (EOMs) for the system are derived
using Kane's method :cite:`kane_dynamics_1985`, which can be further analysed and used
in simulation and optimization tasks.

For a more in-depth academic introduction to SymBRiM and its application in trajectory
tracking problems, refer to :cite:`stienstra_2023_brim`. The rest of this page will
focus on the practical usage of SymBRiM.

Defining a Model Using SymPy
----------------------------
..
    Paragraph: Explain a little about sympy and refer where to find more information.

:mod:`sympy` is a Python library for symbolic mathematics, which has a separate module,
:mod:`sympy.physics.mechanics`, for working with dynamical systems. This module provides
an interface for defining systems and deriving their EOMs. It has an interface that uses
reference frames, points, bodies and joints to describe a system. For a comprehensive
guide on the low-level interface to describe dynamical systems, refer to
:cite:`moore_learn_2022`. An example of defining a simple system, which follows a
similar workflow as SymBRiM, can be found in the docstring of
:class:`sympy.physics.mechanics.system.System`.

Composing a Model
-----------------
..
    Paragraph: Show image and explain the tree structure.

SymBRiM divides the bicycle-rider system into smaller subsystems called submodels. Each
submodel is modeled separately and aggregated to create the complete bicycle-rider. This
results in a tree structure, where the root is the entire bicycle-rider system, and the
bicycle and rider are its children. The bicycle model can be further divided as
illustrated in the image below. This graphical representation shows the division of
the bicycle model into a ground, rear frame, front frame, rear wheel, front wheel and
two tire models.

.. raw:: html

    <svg class="align-center" viewBox="0 0 231.0395 70.457903" width="800px">
        <use xlink:href="../_static/configuration_overview_whipple_default.svg#whippleConfig"></use>
    </svg>


Core Components
---------------
..
    Paragraph: List of core components and their purpose.

SymBRiM makes use of three core components: models, connections and loads groups. The
models are the main components, where each model describes a (sub)system following a
tree structure. The connections can be seen as an utility of parent models to describe a
modular and reusable interaction between submodels. The load groups are predefined sets
of actuators and loads, which are commonly associated with a specific model or
connection.

Defining a Model Using SymBRiM
------------------------------
..
    Paragraph: Create the default Whipple bicycle model using SymBRiM.

The default Whipple bicycle model :cite:`whipple1899stability`, also visualized as a
tree in the image above, can be constructed as follows. The first step is to configure
the model by choosing the components. ::

    from sympy import symbols
    from sympy.physics.mechanics._actuator import TorqueActuator
    from sympy.physics.mechanics import dynamicsymbols
    from symbrim import *

    bicycle = WhippleBicycle("bicycle")
    bicycle.rear_frame = RigidRearFrame("rear_frame")
    bicycle.front_frame = RigidFrontFrame("front_frame")
    bicycle.rear_wheel = KnifeEdgeWheel("rear_wheel")
    bicycle.front_wheel = KnifeEdgeWheel("front_wheel")
    bicycle.rear_tire = NonHolonomicTire("rear_tire")
    bicycle.front_tire = NonHolonomicTire("front_tire")

With the model configured, the next step is to let SymBRiM construct all the relationships
in the model by calling :meth:`~.ModelBase.define_all`. After this, the model can be
exported to a single :class:`sympy.physics.mechanics.system.System` object. ::

    bicycle.define_all()
    system = bicycle.to_system()

The model can be extended by adding an actuator for steering and applying gravity: ::

    g = symbols("g")
    T = dynamicsymbols("T")
    normal = bicycle.ground.get_normal(bicycle.ground.origin)
    system.add_actuators(TorqueActuator(
        T, bicycle.rear_frame.steer_axis,
        bicycle.front_frame.frame, bicycle.rear_frame.frame))
    system.apply_uniform_gravity(-g * normal)

The last step before forming the EOMs is to define which generalized coordinates and
speeds are independent and which are dependent. In this case the generalized coordinate
for the pitch of the rear frame is chosen to be dependent. As dependent generalized
speeds we choose the velocities for the contact point of the rear wheel, the rotation
rate of the rear wheel, and the yaw and pitch rate of the rear frame. ::

    system.q_ind = [*bicycle.q[:4], *bicycle.q[5:]]
    system.q_dep = [bicycle.q[4]]
    system.u_ind = [bicycle.u[3], *bicycle.u[5:7]]
    system.u_dep = [*bicycle.u[:3], bicycle.u[4], bicycle.u[7]]

The following code validates the system on its consistency using
:meth:`sympy.physics.mechanics.system.System.validate_system` and forms the EOMs with
Kane's method. ``CRAMER`` is chosen as the constraint solver, as it provides better
numeric stability. ::

    system.validate_system()
    system.form_eoms(constraint_solver="CRAMER")

See Also
--------
..
    Paragraph: List of related pages and advised locations what to read next.

Here are some useful reference to get started:

- The :ref:`tutorials` page contains a collection of tutorials to get started with
  SymBRiM.
- The `brim-bmd-2023-paper <https://github.com/TJStienstra/brim-bmd-2023-paper>`_
  repository contains a ``src`` directory with various trajectory tracking examples
  discussed in :cite:`stienstra_2023_brim`.
- :cite:`moore_learn_2022` is a great place to learn more about the fundamental modeling
  concepts when using :mod:`sympy.physics.mechanics`.
- The :mod:`sympy.physics.mechanics` module contains several examples and tutorials to
  get familiar with using :mod:`sympy` for modeling dynamical systems.

References
----------

.. bibliography::
