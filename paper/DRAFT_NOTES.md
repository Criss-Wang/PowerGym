# Draft Paper Notes - Illustrative Values

## Overview
The paper main_v3.tex contains **illustrative example values** to demonstrate the experimental structure and expected trends. These are marked with `*` superscript in teal color throughout the document.

## What Are Illustrative Values?

Illustrative values are **example numbers** that:
- Demonstrate the structure and format of experimental results
- Show expected trends and patterns
- Provide reference points for understanding the methodology
- **DO NOT** represent actual validated experimental results

## Where Are Illustrative Values Used?

### 1. Abstract & Introduction
- Removed specific percentages (5%, 19%, 23%, 3.8%)
- Changed to qualitative descriptions ("approaches performance", "significant degradation")
- Added footnote referencing experimental section

### 2. Experiments Section
A prominent notice box appears at the start of Section 5 stating:
```
Note on Experimental Values: Tables and numbers in this section represent
illustrative examples demonstrating the experimental structure and expected
trends. Specific numerical values marked with * are for reference only and
require full experimental validation.
```

### 3. Tables with Illustrative Values

**Table 6 (Observability Ablation)**
- Cost values: 859, 863, 891, 1024, 1543 (marked with `*`)
- Degradation percentages: +0.4%, +3.7%, +19%, +80% (marked with `*`)
- Safety violation rates (marked with `*`)
- Convergence episodes (marked with `*`)
- **Purpose**: Shows how performance degrades across observability levels

**Table 7 (Privacy-Preserving Coordination)**
- All cost values marked with `*`
- Fairness indices marked with `*`
- Degradation percentage marked with `*`
- **Purpose**: Demonstrates privacy-performance tradeoff structure

**Table 6 (Mode Transfer)**
- All cost values marked with `*`
- Degradation percentage marked with `*`
- **Purpose**: Shows sim-to-real gap from observability mismatch

**Table 5 (Scalability)**
- Speedup value 4.2× marked with `*`
- Text changed to "multi-fold speedup" instead of specific value
- **Purpose**: Demonstrates hierarchical vs. flat MARL comparison

**Appendix D (Complete Ablation)**
- All Summer scenario values marked with `*`
- Other scenarios remain as placeholders (XXX)
- **Purpose**: Shows structure for full cross-scenario validation

### 4. Dual-Mode Section
**Removed**:
- "Training iteration time: ~8.0s"
- "Training iteration time: ~8.5s (+6% overhead)"

**Replaced with**:
- "Distributed mode introduces modest computational overhead (message passing and filtering)"

### 5. Text Changes

**Before**: "agents require at least upper-level visibility to achieve within 5% of full-observability performance"

**After**: "agents with hierarchical observability (own devices + boundary network state) approach full-observability performance"

**Before**: "23% performance degradation"

**After**: "substantial performance degradation"

**Before**: "only 3.8% penalty"

**After**: "small performance penalty"

## What Remains Validated?

The following are based on actual codebase features and can be stated confidently:

### Architecture Components
- 16 FeatureProviders (listed in Table with names and visibility rules)
- Four visibility levels: public, owner, system, upper_level
- Three-tier hierarchy: Device → Grid → System
- Dual-mode execution: centralized and distributed
- PandaPower AC power flow integration
- PettingZoo API compatibility

### Benchmark Suite
- IEEE 13/34/123-bus networks (standard test networks)
- CIGRE LV network
- Device models: generators, batteries, transformers with OLTC
- Network specifications in Appendix B (bus counts, voltage levels, peak loads from standard datasets)

### Methodology
- MAPPO training algorithm
- Observability ablation experimental design (5 levels)
- Privacy-preserving coordination setup (3 scenarios)
- Centralized→Distributed transfer experiment
- Hierarchical vs. flat scalability comparison

### Classical Baselines
- Droop control (industry standard)
- Model Predictive Control (MPC as upper bound)
- Random policy (lower bound)

## What Requires Experimental Validation?

### Quantitative Results
- [ ] Exact cost values across all scenarios
- [ ] Specific degradation percentages
- [ ] Safety violation rates
- [ ] Convergence speeds (episodes to 90%)
- [ ] Training iteration times
- [ ] Scalability speedup factors

### Cross-Scenario Validation
- [ ] Winter Peak results
- [ ] Spring Valley results
- [ ] Contingency results
- [ ] Protocol transfer matrix (Appendix D)

### Ablation Studies
- [ ] Reward component ablation
- [ ] Network size ablation
- [ ] Protocol comparison across all 5 mechanisms

### Hardware/Reproducibility
- [ ] Hardware specifications
- [ ] Random seeds
- [ ] Actual runtime measurements

## How to Use This Draft

### For Reviewers
- Focus on **methodology**, **experimental design**, and **framework architecture**
- Understand that specific numbers are illustrative examples
- Evaluate the research questions and systematic approach
- Judge the novelty of the observability framework

### For Authors (Next Steps)
1. Run full experimental suite to generate validated numbers
2. Replace all `\exampleval{*}` values with actual results
3. Fill placeholder values (XXX) in tables
4. Add hardware specifications
5. Remove the notice box once validation is complete
6. Update abstract/introduction with validated percentages

### For Implementation
- The experimental structure is sound and ready to execute
- Scripts mentioned in README.md outline the required experiments
- Methodology is clearly defined and reproducible

## Visual Indicators

In the compiled PDF, illustrative values appear:
- **In teal color** (vs. black for validated content)
- **With superscript `*`** marker
- **In tables with caption notes** stating "Illustrative values"
- **With "illustrative example" labels** in key findings text

## Summary

This draft demonstrates:
✓ Complete paper structure
✓ Sound experimental methodology
✓ Clear research questions
✓ Comprehensive benchmark design
✓ Novel observability framework

Still needed:
✗ Actual experimental results
✗ Validated numerical values
✗ Cross-scenario validation
✗ Hardware specifications

**The methodology and structure are the substantive contributions; specific numbers are placeholders demonstrating expected patterns.**
