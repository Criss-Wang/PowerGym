Devices
=======

.. toctree::
   :hidden:

   Energy Storage <storage>
   Generator <generator>
   Solar PV <solar>
   Transformer <transformer>
   Capacitor Bank <capacitor>

PowerGrid includes models for various distributed energy resources (DER).

Device Types
------------

- :doc:`storage` - Battery energy storage systems
- :doc:`generator` - Diesel generators
- :doc:`solar` - Solar photovoltaic panels
- :doc:`transformer` - Voltage regulation transformers
- :doc:`capacitor` - Reactive power compensation

Device Interface
----------------

All devices implement a common interface:

.. code-block:: python

   class Device:
       def step(self, action, dt):
           """Execute one timestep with given action."""
           pass

       def get_state(self):
           """Return current device state."""
           pass

       def reset(self):
           """Reset device to initial state."""
           pass

Device Models
-------------

Energy Storage System (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Battery model with:

- State of charge (SoC) dynamics
- Charging/discharging efficiency
- Power and energy limits
- Degradation (optional)

Diesel Generator
~~~~~~~~~~~~~~~~

Dispatchable generation with:

- Fuel consumption curve
- Ramp rate limits
- Minimum stable generation
- Start/stop costs

Solar PV
~~~~~~~~

Renewable generation with:

- Irradiance-based generation
- Temperature derating
- Maximum power point tracking
- Curtailment control

Wind Turbine
~~~~~~~~~~~~

Wind generation with:

- Power curve modeling
- Cut-in and cut-out speeds
- Curtailment control

Transformer
~~~~~~~~~~~

Voltage regulation via:

- Tap changing mechanism
- Turn ratio adjustment
- Deadband control
- Switching delays

Capacitor Bank
~~~~~~~~~~~~~~

Reactive power compensation:

- Switched capacitor banks
- Fixed or variable steps
- Voltage-based control

Creating Custom Devices
------------------------

Extend the base ``Device`` class:

.. code-block:: python

   from powergrid.devices.base import Device

   class MyCustomDevice(Device):
       def __init__(self, name, bus, **params):
           super().__init__(name, bus)
           self.params = params

       def step(self, action, dt):
           # Update device state based on action
           self.state = self.dynamics(action, dt)
           return self.state

       def get_state(self):
           return {
               'p_mw': self.state.p_mw,
               'q_mvar': self.state.q_mvar,
               # Custom state variables
           }

       def reset(self):
           self.state = self.initial_state

See Also
--------

- :doc:`/api/devices` - Device API reference
- :doc:`/agents/device_agent` - DeviceAgent that controls devices
