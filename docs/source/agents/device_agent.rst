DeviceAgent
===========

DeviceAgent controls a single distributed energy resource (DER) device.

Overview
--------

A DeviceAgent:

- **Observes** its device state and parent GridAgent signals
- **Executes** control actions on the device
- **Responds** to coordination signals (prices, setpoints)
- **Reports** state back to GridAgent

Supported Devices
-----------------

Energy Storage System (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Battery storage with charging/discharging control.

**Actions:** Charge/discharge power
**Constraints:** SoC limits, power limits, efficiency

Diesel Generator
~~~~~~~~~~~~~~~~

Dispatchable generation with fuel costs.

**Actions:** Real power output
**Constraints:** Min/max generation, ramp rates

Solar PV
~~~~~~~~

Renewable generation with optional curtailment.

**Actions:** Curtailment factor (0-1)
**Constraints:** Available solar irradiance

Load Controller
~~~~~~~~~~~~~~~

Flexible load with demand response capability.

**Actions:** Load shedding fraction
**Constraints:** Critical load requirements

Transformer
~~~~~~~~~~~

Voltage regulation via tap changing.

**Actions:** Tap position
**Constraints:** Tap limits, deadband

Capacitor Bank
~~~~~~~~~~~~~~

Reactive power compensation.

**Actions:** On/off switching
**Constraints:** Switching limits

Configuration
-------------

.. code-block:: python

   from powergrid.agents import DeviceAgent
   from powergrid.devices import StorageDevice

   # Create device
   device = StorageDevice(
       name='ESS1',
       bus=632,
       max_e_mwh=1.0,
       max_p_mw=0.5,
       efficiency=0.95
   )

   # Create agent
   agent = DeviceAgent(
       agent_id='ESS1',
       device=device,
       parent_id='MG1'
   )

Observation Space
-----------------

DeviceAgent observes:

**Device State:**

- Power output (P, Q)
- State of charge (for ESS)
- Temperature (for generators)
- Available capacity

**Local Grid State:**

- Bus voltage
- Frequency
- Time of day

**Messages:**

- Parent GridAgent signals (price, setpoint)

Action Space
------------

Actions depend on device type:

**ESS:**

.. code-block:: python

   action = 0.3  # Charge at 0.3 MW (positive = charge)

**Generator:**

.. code-block:: python

   action = 0.5  # Generate 0.5 MW

**Solar PV:**

.. code-block:: python

   action = 0.0  # No curtailment (1.0 = full curtailment)

**Transformer:**

.. code-block:: python

   action = 1  # Increase tap position by 1 step

Example Usage
-------------

.. code-block:: python

   from powergrid.agents import DeviceAgent
   from powergrid.devices import StorageDevice

   # Create storage device
   device = StorageDevice(
       name='ESS1',
       bus=632,
       max_e_mwh=2.0,
       max_p_mw=0.5
   )

   # Create agent
   agent = DeviceAgent(
       agent_id='ESS1',
       device=device,
       parent_id='MG1'
   )

   # Observe state
   obs = agent.observe(global_state)
   print(f"SoC: {obs.local['soc']}")
   print(f"Voltage: {obs.local['vm_pu']}")

   # Receive price signal from parent
   for msg in obs.messages:
       if 'price' in msg.content:
           price = msg.content['price']
           # Optimize local action based on price
           if price < 40:
               action = 0.5  # Charge when cheap
           elif price > 80:
               action = -0.5  # Discharge when expensive
           else:
               action = 0.0

   # Execute action
   agent.act(action)

See Also
--------

- :doc:`grid_agent` - Parent GridAgent
- :doc:`/devices/index` - Device models
- :doc:`/api/agents` - API reference
