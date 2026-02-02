# HERON: NeurIPS 2025 D&B Submission Plan

## Strategic Positioning

**Track:** NeurIPS 2025 Datasets & Benchmarks
**Category:** Framework/Infrastructure Paper (Path A)
**Positioning:** Principled framework with demonstrated utility
**Target Score:** 7+/10 (Clear Accept)

---

## Core Framing

> "HERON is a principled framework that shifts from environment-centric synchronous stepping to agent-paced event-driven execution, enabling simulation of realistic CPS timing constraints while providing infrastructure for studying partial observability and information asymmetry in multi-agent systems."

---

## Differentiation from PettingZoo

**The core reviewer objection:** "Is this PettingZoo + engineering effort, or a genuinely new abstraction?"

**Our answer:** PettingZoo standardizes the **environment API** (reset, step, observe, act). HERON standardizes **what happens inside the environment**:

| Aspect | PettingZoo | HERON |
|--------|------------|-------|
| Abstraction level | Env ↔ Algorithm interface | Internal env structure |
| Execution model | Fixed (synchronous step) | Configurable (sync or event-driven) |
| Information structure | Black box (env decides) | First-class variable (visibility levels) |
| Coordination | Not addressed | Composable protocols |
| Agent model | Stateless policy function | Stateful entity with timing |

**Key argument:** Event-driven execution CANNOT be achieved by wrapping—it requires changing the fundamental step loop. This is architectural, not engineering.

**Required evidence:** Head-to-head implementation comparison in Appendix:

| Task | PettingZoo + Wrappers | HERON | Reduction |
|------|----------------------|-------|-----------|
| Visibility ablation (4 levels) | 4 × 80 = 320 lines | 4 config strings | 80x |
| Protocol swap (3 protocols) | 3 × 120 = 360 lines | 3 config strings | 120x |
| Event-driven evaluation | 200 lines (rewrite step loop) | 1 flag | ∞ (impossible to wrap) |

---

## Ecosystem Positioning (Beyond PettingZoo)

**Reviewer concern:** "The paper only compares to PettingZoo. What about other MARL frameworks?"

**Required comparison table:**

| Framework | Abstraction Level | Event-Driven | Visibility Control | Protocol Composition | CPS Focus |
|-----------|-------------------|--------------|--------------------|--------------------|-----------|
| **HERON** | Internal env structure | ✅ Native | ✅ First-class | ✅ Swappable | ✅ Yes |
| PettingZoo | Env ↔ Algo interface | ❌ No | ❌ Manual | ❌ No | ❌ No |
| EPyMARL | Algorithm benchmarking | ❌ No | ❌ No | ❌ No | ❌ No |
| MARLlib | Algorithm library | ❌ No | ❌ No | ❌ No | ❌ No |
| SMAC/SMACv2 | Domain-specific (StarCraft) | ❌ No | Partial | ❌ No | ❌ No |
| MALib | Population training | ❌ No | ❌ No | ❌ No | ❌ No |

**Key argument:** These frameworks focus on algorithm benchmarking or specific domains. HERON addresses *environment construction* for CPS with timing constraints—orthogonal and complementary.

**Implementation evidence (Appendix):**
- [ ] Implement power microgrid in EPyMARL: measure LOC, time, feature gaps
- [ ] Attempt event-driven in MARLlib: document impossibility
- [ ] Show HERON environments can export to PettingZoo API for algorithm compatibility

---

## Four Contributions

### 1. Agent-Centric Architecture

**Claim:** Agents as first-class citizens with internal state, timing, and observability.

**Language:**
> "Unlike environment-centric frameworks where agents are passive policy containers, HERON treats agents as first-class entities with configurable internal state, timing, and observability."

### 2. Dual Execution Modes (HEADLINE CONTRIBUTION)

**Claim:** Synchronous training + agent-paced event-driven deployment validation.

**Language:**
> "HERON provides dual execution modes: synchronous training for efficient policy learning, and agent-paced event-driven execution for validating policies under realistic CPS timing constraints."

**Required validation:** Experiments with CPS-calibrated timing distributions:

| Domain | Timing Source | Distribution | Parameters |
|--------|---------------|--------------|------------|
| Power (SCADA) | IEEE Std 2030-2011 | LogNormal | μ=2s, σ=0.8s |
| Power (PMU) | NASPI 2018 Report | Deterministic | 16.67ms (60Hz) |
| Traffic | NTCIP 1202 | Uniform | [100ms, 500ms] |

### 3. Systematic Information Partitioning (CALIBRATED)

**Valid claim:** FeatureProviders with visibility levels enable controlled observability ablation.

**Invalid claim:** ~~Privacy-preserving RL~~

**Language:**
> "ProxyAgents mediate state access, applying visibility-based filtering to enable systematic study of partial observability and information asymmetry."

### 4. Composable Coordination Protocols

**Claim:** Vertical and horizontal protocols are swappable at configuration time.

**Language:**
> "HERON's protocol system separates coordination mechanism from agent logic, enabling researchers to compare different protocols as experimental variables."

---

## Paper Abstract

> We present HERON, a framework that shifts from environment-centric synchronous stepping to agent-paced event-driven execution, enabling simulation of realistic CPS timing constraints while providing infrastructure for studying partial observability and information asymmetry in multi-agent systems. Our key contributions are:
>
> 1. **Agent-centric architecture**: Unlike environment-centric frameworks where agents are passive policy containers, HERON treats agents as first-class entities with configurable internal state, timing, and observability.
>
> 2. **Dual execution modes**: Synchronous training for efficient policy learning, and agent-paced event-driven execution for validating policies under realistic CPS timing constraints—bridging the sim-to-real gap for distributed systems.
>
> 3. **Systematic information partitioning**: FeatureProviders with visibility levels enable controlled study of partial observability and information asymmetry, making observability a first-class experimental variable.
>
> 4. **Composable coordination protocols**: Vertical and horizontal protocols are swappable at configuration time, enabling systematic comparison of coordination mechanisms.
>
> We demonstrate HERON's utility through power systems and traffic network case studies, showing consistent abstractions across domains.

### Explicit Non-Claims

> "HERON provides infrastructure for studying information constraints in MARL. It is not a privacy-preserving framework in the differential privacy or cryptographic sense—agents may still infer information beyond their direct observations through learning."

---

## Case Studies

### Power Systems (Primary)

| Component | Count | Status |
|-----------|-------|--------|
| FeatureProviders | 14 | ✅ Done |
| Protocols | 4 (Setpoint, PriceSignal, P2P, Consensus) | ✅ Done |
| Network size | 5 microgrids | ✅ Done |

### Traffic Networks (MUST Match Power Maturity — Hard Requirement)

| Component | Target | Power Equivalent | Status |
|-----------|--------|------------------|--------|
| FeatureProviders | 14 | 14 FeatureProviders | [ ] TODO |
| Protocols | 4 | 4 (Setpoint, PriceSignal, P2P, Consensus) | [ ] TODO |
| Network size | 25 intersections (5×5 grid) | 5 microgrids | [ ] TODO |
| Agents | 25 | 25 (5 microgrids × 5 devices) | [ ] TODO |

**Traffic FeatureProviders (14 total, matching power domain count):**

| Visibility | Provider | Power Equivalent |
|------------|----------|------------------|
| owner | QueueLengthProvider | BatterySOCProvider |
| owner | PhaseDurationProvider | GeneratorOutputProvider |
| owner | ThroughputProvider | LoadDemandProvider |
| owner | WaitTimeProvider | VoltageProvider |
| owner | CyclePositionProvider | FrequencyProvider |
| upper_level | NeighborQueueProvider | NeighborSOCProvider |
| upper_level | CorridorFlowProvider | MicrogridPowerProvider |
| upper_level | CoordinationStateProvider | CoordinationStateProvider |
| system | NetworkDemandProvider | GridDemandProvider |
| system | IncidentProvider | OutageProvider |
| system | GlobalCongestionProvider | SystemFrequencyProvider |
| system | PeakHourIndicatorProvider | PeakDemandProvider |
| public | WeatherProvider | WeatherProvider |
| public | TimeOfDayProvider | TimeOfDayProvider |

**Traffic Protocols (4 total, matching power domain count):**

| Protocol | Type | Power Equivalent |
|----------|------|------------------|
| FixedTimingProtocol | Vertical (top-down) | SetpointProtocol |
| AdaptiveOffsetProtocol | Vertical (feedback) | PriceSignalProtocol |
| GreenWaveProtocol | Horizontal (corridor) | P2PEnergyProtocol |
| ConsensusTimingProtocol | Horizontal (distributed) | ConsensusProtocol |

**Parity Checklist (ALL must be ✅ before submission):**
- [ ] Same number of FeatureProviders (14)
- [ ] Same number of protocols (4)
- [ ] Same visibility level distribution (5 owner, 3 upper_level, 4 system, 2 public)
- [ ] Comparable network complexity (25 intersections ≈ 25 power devices)
- [ ] All experiments run on BOTH domains with comparable results

---

## Experiments

### Required Experiments (12 total)

| # | Experiment | Domain | Purpose | Parity Required |
|---|------------|--------|---------|-----------------|
| 1 | Visibility ablation | Power | Information structure matters | ✅ Paired with #2 |
| 2 | Visibility ablation | Traffic | Cross-domain consistency | ✅ Paired with #1 |
| 3 | Protocol comparison | Power | Protocols are swappable | ✅ Paired with #4 |
| 4 | Protocol comparison | Traffic | Cross-domain consistency | ✅ Paired with #3 |
| 5 | Event-driven gap (5 timing configs) | Power | CPS-calibrated validation | ✅ Paired with #6 |
| 6 | Event-driven gap (5 timing configs) | Traffic | Cross-domain consistency | ✅ Paired with #5 |
| 7 | Algorithm comparison (4 algos) | Power | Algorithm-agnostic | ✅ Paired with #8 |
| 8 | Algorithm comparison (4 algos) | Traffic | Cross-domain consistency | ✅ Paired with #7 |
| 9 | Scalability (10-2000 agents) | Both | Scaling characteristics | — |
| 10 | MessageBroker overhead | Synthetic | Abstraction cost <5% | — |
| 11 | ~~User study~~ | ~~Both~~ | ~~Not required for D&B~~ | — |
| 12 | Framework comparison (PettingZoo, EPyMARL) | Power | Architectural differentiation | — |

**Cross-Domain Consistency Check:**
For experiments 1-8, the *relative ordering* of results must hold across both domains. E.g., if `system > upper_level > owner` for visibility in power, the same ordering must appear in traffic. This is the key evidence for domain-agnostic abstractions.

### Algorithm Diversity

| Algorithm | Category | Status |
|-----------|----------|--------|
| MAPPO | Policy gradient | ✅ Done |
| IPPO | Independent | ✅ Done |
| QMIX | Value decomposition | [ ] TODO |
| TarMAC | Communication | [ ] TODO |

**Key result:** Relative ordering (system > upper_level > owner) should hold across all algorithms.

### Event-Driven Timing Configurations

| Config | Distribution | Parameters | Purpose |
|--------|--------------|------------|---------|
| Baseline | Synchronous | All agents tick simultaneously | Training condition |
| Uniform | Uniform(0, τ_max) | τ_max ∈ {0.5s, 1s, 2s, 4s} | Sensitivity |
| SCADA | LogNormal(μ, σ) | μ=2s, σ=0.8s | Real-world calibration |
| Jitter | Gaussian(0, σ) + base | σ ∈ {0.1s, 0.5s, 1s} | Communication noise |
| Heterogeneous | Per-agent different τ | Fast/slow mix | Realistic deployment |

**Metrics:** Performance degradation curve, critical delay threshold, recovery with fine-tuning.

### Scalability (Target: 1000+ agents)

**Reviewer concern:** "500 agents is modest by 2026 standards."

| Agent Count | Step Time (sync) | Step Time (event) | Memory | Target |
|-------------|------------------|-------------------|--------|--------|
| 10 | TBD | TBD | TBD | <1ms |
| 50 | TBD | TBD | TBD | <5ms |
| 100 | TBD | TBD | TBD | <10ms |
| 500 | TBD | TBD | TBD | <50ms |
| 1000 | TBD | TBD | TBD | <100ms |
| 2000 | TBD | TBD | TBD | <250ms |

**Bottleneck Analysis & Mitigations:**

| Component | Naive Complexity | Mitigation | Target Complexity |
|-----------|------------------|------------|-------------------|
| MessageBroker | O(n²) broadcast | Topic-based routing, lazy eval | O(n × subscribers) |
| EventScheduler | O(n log n) | Heap-based priority queue | O(log n) per event |
| FeatureProvider | O(n) per query | Caching, batch computation | O(1) amortized |
| Visibility filtering | O(n × features) | Precomputed visibility masks | O(features) |

**Comparison with Large-Scale MARL:**
- Mean-field games handle 10K+ but assume homogeneous agents
- HERON targets heterogeneous CPS agents with distinct timing—1000-2000 is realistic for power/traffic
- Report both absolute performance AND scaling coefficient (e.g., "1.8x slowdown for 2x agents")

**Required plots:**
- [ ] Log-log scaling plot (agents vs. step time)
- [ ] Memory scaling plot
- [ ] Breakdown by component (MessageBroker, EventScheduler, FeatureProvider)

### User Study (OPTIONAL — Not Required for D&B)

**Precedent:** PettingZoo, RLlib, Gymnasium, MARLlib, EPyMARL, SMAC — none include user studies. User studies are typical for CHI/CSCW, not NeurIPS D&B.

**If included (nice-to-have, not required):**
- N=3-5 informal pilot with qualitative feedback
- Focus on "pain points" and "aha moments", not statistical claims
- Can be mentioned in 1-2 sentences, not a full section

**Alternative evidence (stronger for D&B):**
- Clear API documentation with tutorials
- Working examples in repository
- LOC comparison tables (already planned)
- Video walkthrough (5 min)

---

## Implementation Plan

### Phase 1: Framework Polish + Rigorous Framework Comparison (Week 1-2)

**Objective:** Provide evidence beyond LOC counts—measure researcher effort and architectural limitations.

- [ ] Complete API documentation
- [ ] **PettingZoo comparison (rigorous):**
  - [ ] Implement PettingZoo visibility wrapper (measure: LOC, time, bugs encountered)
  - [ ] Implement PettingZoo protocol wrapper (measure: LOC, time, bugs encountered)
  - [ ] Attempt PettingZoo event-driven loop → document *why* it fails (not just that it's hard)
  - [ ] Record video of implementation attempts for supplementary material
- [ ] **EPyMARL/MARLlib comparison:**
  - [ ] Attempt to implement microgrid env in EPyMARL
  - [ ] Document missing abstractions (visibility, protocols, event-driven)
- [ ] **Interoperability demonstration:**
  - [ ] Show HERON env exporting to PettingZoo API
  - [ ] Run PettingZoo-compatible algorithm on HERON env
- [ ] Create comparison table with *qualitative* dimensions (not just LOC)
- [ ] Ensure all tests pass

### Phase 2: Traffic Domain Expansion (Week 2-4) — CRITICAL PATH

**Objective:** Achieve exact parity with power domain. This is the highest-risk item.

- [ ] Implement **14 FeatureProviders** (matching power domain count):
  - [ ] 5 owner-level: QueueLength, PhaseDuration, Throughput, WaitTime, CyclePosition
  - [ ] 3 upper_level: NeighborQueue, CorridorFlow, CoordinationState
  - [ ] 4 system-level: NetworkDemand, Incident, GlobalCongestion, PeakHourIndicator
  - [ ] 2 public: Weather, TimeOfDay
- [ ] Implement **4 protocols** (matching power domain count):
  - [ ] FixedTimingProtocol (vertical, top-down)
  - [ ] AdaptiveOffsetProtocol (vertical, feedback)
  - [ ] GreenWaveProtocol (horizontal, corridor)
  - [ ] ConsensusTimingProtocol (horizontal, distributed)
- [ ] Create **5×5 grid network** (25 intersections, matching 25 power devices)
- [ ] Verify same base classes work (no traffic-specific hacks)
- [ ] Run smoke tests: visibility ablation, protocol swap on traffic domain
- [ ] **Parity checkpoint:** Side-by-side comparison table showing identical structure

### Phase 3: Event-Driven Validation (Week 3-4)

- [ ] Gather IEEE 2030, NTCIP, NASPI references
- [ ] Implement configurable delay distributions
- [ ] Run 5 timing configurations on both domains
- [ ] Plot degradation curves
- [ ] Identify critical thresholds

### Phase 4: Algorithm Diversity (Week 4-5)

- [ ] Integrate QMIX
- [ ] Integrate TarMAC
- [ ] Run visibility ablation for all 4 algorithms
- [ ] Verify consistent patterns

### Phase 5: Scalability + Documentation (Week 5-6)

**Scalability:
- [ ] Benchmark 10, 50, 100, 500, 1000, 2000 agents
- [ ] Profile bottlenecks: MessageBroker, EventScheduler, FeatureProvider
- [ ] Implement mitigations if needed (topic-based routing, caching)
- [ ] Create log-log scaling plots
- [ ] Compare scaling coefficient to mean-field baselines

**Documentation (matches PettingZoo/RLlib standard):**
- [ ] API reference documentation
- [ ] Quick-start tutorial (power domain)
- [ ] Quick-start tutorial (traffic domain)
- [ ] 5-min video walkthrough

### Phase 6: Paper + Submission (Week 5-7)

- [ ] Write paper with all results
- [ ] Add appendices (PettingZoo comparison, user study)
- [ ] Clean GitHub repo + ReadTheDocs
- [ ] Video tutorial (5 min)

---

## Language Guidelines

### Use

| Context | Phrasing |
|---------|----------|
| Observability | "partial observability", "information asymmetry" |
| Execution | "agent-paced event-driven", "CPS timing constraints" |
| Architecture | "shifts from environment-centric to agent-centric" |
| PettingZoo | "different abstraction level", "complements" |

### Avoid

| Context | Phrasing |
|---------|----------|
| Privacy | ~~"privacy-preserving"~~, ~~"privacy guarantees"~~ |
| Claims | ~~"we discover"~~, ~~"our finding"~~ |
| Competition | ~~"better than PettingZoo"~~ |

---

## Required References (Do Not Miss)

**MARL Frameworks & Benchmarks:**
- Terry et al., "PettingZoo: Gym for Multi-Agent Reinforcement Learning" (JMLR 2021)
- Papoudakis et al., "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks" (NeurIPS 2021)
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022)
- Hu et al., "MARLlib: A Scalable Multi-agent Reinforcement Learning Library" (JMLR 2023)

**Communication & Coordination:**
- Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" (NeurIPS 2016)
- Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation" (NeurIPS 2016)
- Das et al., "TarMAC: Targeted Multi-Agent Communication" (ICML 2019)

**CPS & Timing Standards:**
- IEEE Std 2030-2011: Guide for Smart Grid Interoperability
- NASPI 2018: Synchrophasor Technology and PMU Performance
- NTCIP 1202: Object Definitions for Actuated Signal Controllers

**Power Systems RL:**
- Zhang et al., "Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms" (2021)
- Wang et al., "Review of Deep Reinforcement Learning for Power System Applications" (2020)

**Traffic Signal Control:**
- Wei et al., "A Survey on Traffic Signal Control Methods" (2019)
- Zheng et al., "Learning Phase Competition for Traffic Signal Control" (CIKM 2019)

---

## Go/No-Go Criteria

**Submit if ALL true:**

- [ ] **Framework comparison** shows architectural difference with documented evidence:
  - [ ] Event-driven impossible to wrap (with explanation of *why*, not just LOC)
  - [ ] Comparison includes EPyMARL/MARLlib, not just PettingZoo
  - [ ] Interoperability demonstrated (HERON → PettingZoo API export)
- [ ] Event-driven uses CPS-calibrated timing with IEEE/NTCIP citations
- [ ] 4 algorithms tested with consistent visibility pattern across BOTH domains
- [ ] **Traffic domain parity achieved:**
  - [ ] 14 FeatureProviders (same as power)
  - [ ] 4 protocols (same as power)
  - [ ] 25 intersections / 25 agents (comparable to power)
  - [ ] Same visibility level distribution
- [ ] Clear documentation + tutorials + video walkthrough
- [ ] Scalability benchmarked up to **1000+ agents** with bottleneck analysis
- [ ] No "privacy-preserving" claims

**Do NOT submit if ANY true:**

- [ ] Traffic domain has fewer FeatureProviders or protocols than power
- [ ] No documentation or working examples in repository
- [ ] Only compared to PettingZoo (must include EPyMARL or MARLlib)
- [ ] Only 2 algorithms tested
- [ ] Event-driven not calibrated to real CPS timing standards
- [ ] Scalability only tested up to 500 agents

---

## Acceptance Probability

| State | Probability | Key Risks |
|-------|-------------|-----------|
| Before mitigations | 30-40% | PettingZoo differentiation, incomplete traffic |
| With ecosystem comparison (not just PettingZoo) | 50-60% | Incomplete traffic domain |
| With traffic parity achieved | 65-75% | Scalability modest |
| With scalability 1000+ agents | 75-85% | No real-world validation |
| **With all addressed** | **80-85%** | Residual: novelty perception |

---

## Optional: Real-World Validation (Stretch Goal)

**Reviewer note:** "Any pilot deployment data, even on a microgrid testbed, would strengthen claims substantially."

**Options (in order of feasibility):**

| Option | Effort | Impact | Feasibility |
|--------|--------|--------|-------------|
| Hardware-in-the-loop simulation | Medium | High | Check if lab has OPAL-RT or similar |
| Collaboration with utility/DOE lab | High | Very High | Requires existing relationship |
| Published testbed data integration | Low | Medium | Use IEEE test feeders with real timing data |
| Qualitative interviews with practitioners | Low | Medium | 3-5 interviews with grid operators |

**Minimum viable evidence:**
- [ ] Cite existing CPS testbed papers that use similar timing distributions
- [ ] Show that HERON's timing configs match published measurement studies
- [ ] Include "Future Work" section on hardware-in-the-loop integration

**If time permits:**
- [ ] Run HERON with timing traces from real SCADA logs (anonymized, if available)
- [ ] Partner with traffic lab for SUMO-HERON integration validation
