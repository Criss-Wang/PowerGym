# PowerGrid 2.0: Design Documentation

This folder contains design documentation for the PowerGrid 2.0 multi-agent reinforcement learning platform.

---

## Document Index

### 📋 Core Documents

1. **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** ⭐ **START HERE**
   - **What it contains**: Complete description of the implemented architecture
   - **When to read**: Understanding current codebase structure
   - **Status**: ✅ Up-to-date (2025-11-10)
   - **Audience**: Developers, researchers, contributors

2. **[architecture_diagrams.md](architecture_diagrams.md)**
   - **What it contains**: Visual architecture diagrams (Mermaid)
   - **When to read**: Understanding system design at a glance
   - **Status**: 🔄 Being updated to match implementation
   - **Audience**: All users

3. **[proposal.md](proposal.md)**
   - **What it contains**: Original design proposal and roadmap
   - **When to read**: Understanding design motivation and future plans
   - **Status**: ⚠️ Historical reference (not fully implemented)
   - **Audience**: Project managers, stakeholders

4. **[TODO.md](TODO.md)**
   - **What it contains**: Development tasks and progress tracking
   - **When to read**: Contributing to the project
   - **Status**: 🔄 Active development tracker
   - **Audience**: Contributors, developers

---

## Quick Navigation

### For New Users
2. Check [architecture_diagrams.md](architecture_diagrams.md) for visual overview
3. Review usage examples in IMPLEMENTATION_STATUS.md

### For Contributors
1. Check [TODO.md](TODO.md) for open tasks
3. Read [proposal.md](proposal.md) for long-term vision

### For Researchers
2. Check [architecture_diagrams.md](architecture_diagrams.md) for agent hierarchy
3. Review protocol system in IMPLEMENTATION_STATUS.md §1.4

---

## Key Architectural Concepts

### Agent Hierarchy (2 Levels Implemented)
```
NetworkedGridEnv (PettingZoo)
├── GridAgent (Level 2): Microgrid controllers
│   └── DeviceAgent (Level 1): DERs (Gen, Storage, RES)
```

### Protocol System (Dual-Type)
- **Vertical Protocols** (Agent-Owned): Parent → Child coordination
  - `PriceSignalProtocol`, `SetpointProtocol`
- **Horizontal Protocols** (Environment-Owned): Peer ↔ Peer coordination
  - `PeerToPeerTradingProtocol`, `ConsensusProtocol`

### Core Modules
- **`core/`**: State, Action, Protocols
- **`agents/`**: Agent, DeviceAgent, GridAgent
- **`devices/`**: Generator, ESS, Grid
- **`features/`**: ElectricalBasePh, StorageBlock, StatusBlock, etc.
- **`envs/`**: NetworkedGridEnv (PettingZoo)
- **`networks/`**: IEEE 13/34/123, CIGRE LV

---

## Implementation Status Summary

### ✅ Completed (Phase 1-2)
- Agent abstraction layer
- Feature-based device state system
- Vertical + horizontal protocol system
- DeviceAgent implementations (Generator, ESS, Grid)
- GridAgent with centralized coordination
- PettingZoo environment integration
- PandaPower backend for power flow

### 🔄 In Progress
- Example environments and documentation
- Additional device types (RES, Shunt, Compensation)
- Decentralized GridAgent coordination

### ⏸️ Deferred
- SystemAgent (Level 3 - ISO/market operator)
- YAML configuration system
- Plugin system for custom devices
- Pre-loaded datasets (CAISO/ERCOT)
- Advanced protocols (ADMM, droop control)

---

## Document Status Legend

- ✅ **Up-to-date**: Reflects current implementation
- 🔄 **In progress**: Being actively updated
- ⚠️ **Historical**: Original proposal, not fully implemented
- ⏸️ **Deferred**: Planned for future releases

---

## Contributing to Documentation

### When to Update Docs
1. **After implementing a new feature**: Update IMPLEMENTATION_STATUS.md
2. **After major refactoring**: Update architecture_diagrams.md
3. **When starting new work**: Update TODO.md

### Documentation Style
- Use clear, concise language
- Include code examples for complex concepts
- Add diagrams for architectural changes
- Keep IMPLEMENTATION_STATUS.md as single source of truth

---

## External References

### Related Documentation
- Main README: `../../README.md`
- API Reference: `../api/` (auto-generated from docstrings)
- Tutorials: `../tutorials/`
- Examples: `../../examples/`

### Research Papers
- TBD: PowerGrid 2.0 paper (in preparation)

---

## Changelog

### 2025-11-10
- Created IMPLEMENTATION_STATUS.md (comprehensive architecture doc)
- Updated README.md with clear navigation guide
- Marked proposal.md as historical reference

### 2025-10-19
- Updated proposal.md with refined protocol system design

### 2025-10-07
- Created initial proposal.md with 3-month roadmap

---

**Maintainer**: PowerGrid Development Team
**Last Updated**: 2025-11-10
**Next Review**: 2025-12-01
