Use Cases
=========

PowerGrid 2.0 enables various power grid control and optimization use cases.

Microgrid Energy Management
----------------------------

Optimize distributed energy resources (DER) in microgrids including batteries, solar panels, and generators.

- **Goal**: Minimize operating costs while maintaining grid stability
- **Agents**: GridAgent per microgrid, DeviceAgents per DER
- **Protocols**: Price signals or setpoint control

Peer-to-Peer Energy Trading
----------------------------

Enable local energy markets where microgrids trade surplus energy.

- **Goal**: Reduce reliance on main grid through local trading
- **Agents**: GridAgents represent market participants
- **Protocols**: P2P trading protocol for market clearing

Voltage Regulation
------------------

Maintain voltage levels across the distribution network within safe limits.

- **Goal**: Prevent over/under voltage violations
- **Agents**: GridAgents coordinate transformer tap changers and capacitor banks
- **Protocols**: Consensus or centralized coordination

Frequency Control
-----------------

Coordinate distributed generation to maintain grid frequency at 60 Hz.

- **Goal**: Balance generation and load in real-time
- **Agents**: GridAgents control generators and storage
- **Protocols**: Droop control or consensus-based frequency regulation

Integration with Renewables
----------------------------

Handle uncertainty from solar and wind generation.

- **Goal**: Maximize renewable penetration while ensuring reliability
- **Agents**: GridAgents forecast and schedule resources
- **Protocols**: Rolling horizon optimization with price signals

Emergency Response
------------------

Coordinate islanding and black start procedures during outages.

- **Goal**: Maintain critical loads during grid disturbances
- **Agents**: GridAgents prioritize load shedding decisions
- **Protocols**: Hierarchical emergency protocols
