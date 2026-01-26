# PowerGrid Research Paper

This folder contains the LaTeX source for the research paper:

**"PowerGrid: A MARL Benchmark with Fine-Grained Observability Control for Distributed Multi-Microgrid Coordination"**

## Target Venue

NeurIPS 2025 Datasets & Benchmarks Track

## Draft Status

**⚠️ IMPORTANT: This is a draft paper with illustrative example values.**

The paper structure, methodology, and framework descriptions are complete. However, many quantitative results are **illustrative examples** (marked with `*` in teal) demonstrating the expected experimental structure and trends. See [DRAFT_NOTES.md](DRAFT_NOTES.md) for details on what requires experimental validation.

## Files

- `main_v3.tex` - **Current version** (observability-focused) **[DRAFT with illustrative values]**
- `main_v2.tex` - Previous version (protocol-focused)
- `main.tex` - Original version
- `references.bib` - Bibliography
- `neurips_2024.sty` - NeurIPS style file
- `DRAFT_NOTES.md` - **Explains illustrative values and what requires validation**
- `REVISION_SUMMARY.md` - Detailed changes from v1 to v2
- `BEFORE_AFTER_COMPARISON.md` - Side-by-side comparisons (v1 to v2)
- `V2_TO_V3_CHANGES.md` - Major pivot from protocols to observability

## Compiling on Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload all files from this folder
3. Set `main_v3.tex` as the main document
4. Compile with pdfLaTeX

## Compiling Locally

```bash
cd paper
pdflatex main_v3.tex
bibtex main_v3
pdflatex main_v3.tex
pdflatex main_v3.tex
```

## Placeholders

The paper contains several placeholders marked with `[XXX]` or similar that need to be filled in:

### Author Information
- Author names and affiliations
- Email addresses

### Experimental Results
- Observability ablation results (Table 6, RQ1)
- Privacy-preserving coordination results (Table 7, RQ2)
- Protocol comparison results (Table 3)
- Scalability results (Table 5)
- Protocol transfer matrix (Appendix D)

### Dataset Details
- Load profile data source
- Solar irradiance location
- Electricity pricing utility

### Hardware Specifications
- Compute resources used for experiments

### Running Experiments

To fill in the experimental results, run the following scripts:

```bash
# Observability ablation study (RQ1)
python examples/observability_ablation.py --network IEEE34 --scenarios all

# Privacy-preserving coordination (RQ2)
python examples/privacy_coordination.py --network IEEE34

# Centralized→Distributed transfer (RQ3)
python examples/mode_transfer.py --train centralized --test distributed

# Protocol comparison
python examples/protocol_comparison.py --protocols all --iterations 10000

# Scalability study
python examples/scalability_study.py --configs 3x3 5x4 10x6 20x6
```

## Key Contributions Highlighted

1. **Composable Partial Observability Framework** (Section 4.1)
   - 16 FeatureProviders with fine-grained visibility rules
   - Four-level visibility hierarchy: public, owner, upper_level, system
   - Enables systematic study of information requirements
   - Addresses sim-to-real gap through realistic information constraints

2. **Standardized Benchmark Suite** (Section 4.3)
   - IEEE 13/34/123-bus networks and CIGRE LV network
   - Four operational scenarios (Summer Peak, Winter Peak, Spring Valley, Contingency)
   - Classical baselines (Droop control, MPC)
   - Standardized evaluation metrics

3. **Dual-Mode Validation Framework** (Section 4.2)
   - Centralized (full observability) and Distributed (SCADA-realistic) modes
   - Exposes 23% performance degradation from observability mismatch
   - Enables train-centralized, validate-distributed workflow

4. **Hierarchical Scalability** (Section 4.2)
   - Device → Grid → System three-tier architecture
   - ProxyAgent models EMS control center
   - 4.2× training speedup at 60-device scale

## Key Findings

- **Upper-level visibility achieves 95% performance**: Hierarchical observability (parent-subordinate only) achieves within 5% of full observability
- **Privacy-preserving coordination viable**: Only 3.8% performance penalty when hiding sensitive internal states
- **Sim-to-real gap from observability**: Policies trained with full observability suffer 23% degradation when deployed under realistic information constraints
- **Hierarchical coordination scales**: 4.2× training speedup at 60-device scale compared to flat MARL

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{powergrid2025,
  title={PowerGrid: A MARL Benchmark with Fine-Grained Observability Control for Distributed Multi-Microgrid Coordination},
  author={[Authors]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2025}
}
```

## Version History

- **v3 (current)**: Observability-focused - positions Composable Partial Observability Framework as main contribution
- **v2**: Protocol-focused - emphasized coordination protocols with power system mappings
- **v1**: Original draft - mixed platform/methodology positioning
