Optimization
============

Power flow optimization solvers using Mixed-Integer Second-Order Cone Programming (MISOCP).

.. note::

   The optimization modules require external data files that are not included in the repository.
   See the source files for implementation details.

MISOCP Solver
-------------

General MISOCP power flow solver for optimal power dispatch.

**Module:** ``powergrid.optimization.misocp``

Key classes and functions:

- Optimal power flow formulation using MISOCP
- Branch flow model constraints
- Voltage and thermal limit constraints

MISOCP IEEE 123
---------------

IEEE 123-bus specific MISOCP solver.

**Module:** ``powergrid.optimization.misocp_ieee123``

Specialized solver for the IEEE 123-bus test network.
