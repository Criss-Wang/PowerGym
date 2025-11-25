IEEE 13-Bus Test Feeder
=======================

The IEEE 13-bus test feeder is a standard distribution system benchmark.

Network Characteristics
-----------------------

- **Buses**: 13
- **Voltage**: 4.16 kV line-to-line
- **Loads**: Residential and commercial
- **Lines**: Overhead and underground
- **Regulators**: Voltage regulator at substation
- **Capacitors**: Shunt capacitor banks

Microgrid Configuration
-----------------------

The default configuration includes two microgrids:

**MG1 (Buses 632-633-634)**

- 1x Diesel Generator (0.5 MW)
- 1x Battery Storage (1.0 MWh)
- 1x Solar PV (0.3 MW peak)

**MG2 (Buses 671-680-684)**

- 1x Diesel Generator (0.3 MW)
- 1x Battery Storage (0.5 MWh)

Usage
-----

.. code-block:: python

   from powergrid.envs.multi_agent import IEEE13MGEnv

   env = IEEE13MGEnv(
       mode='distributed',
       num_microgrids=2,
       timestep_minutes=15
   )

   observations, infos = env.reset()

Topology Diagram
----------------

.. code-block:: text

   650 ──[Reg]── 632 ─── 633 ─── 634
                  │
                 671 ─── 680 ─── 684
                  │
                 645 ─── 646
                  │
                 692 ─── 675 ─── 611

   MG1: Buses 632-633-634
   MG2: Buses 671-680-684

Challenges
----------

- **Unbalanced loads**: Three-phase imbalance
- **Voltage regulation**: Maintain voltage within ±5%
- **Peak shaving**: Reduce demand charges
- **Coordination**: Trade energy between MG1 and MG2
