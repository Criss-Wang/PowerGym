# Installation

This guide walks you through installing PowerGrid 2.0 and its dependencies.

---

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

---

## Quick Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-lab/powergrid.git
cd powergrid

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Option 2: From PyPI (Coming Soon)

```bash
pip install powergrid
```

---

## Core Dependencies

PowerGrid 2.0 requires the following core packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >=1.21.0, <2.0 | Numerical computation |
| `pandas` | >=2.3.1 | Data handling |
| `gymnasium` | >=1.0.0 | RL environment interface |
| `pandapower` | >=3.1.0 | AC power flow simulation |
| `pettingzoo` | >=1.24.0 | Multi-agent interface |

These are automatically installed when you run `pip install -e .`

---

## Multi-Agent RL Dependencies

For training multi-agent policies, install:

```bash
pip install ray[rllib]==2.9.0
pip install torch>=2.0.0
```

**Note**: Ray RLlib requires `numpy <2.0`, which is already pinned in requirements.

---

## Optional Dependencies

### Optimization (Recommended)

```bash
pip install pyscipopt>=4.0.0
```

Enables advanced optimization solvers for coordination protocols.

### Visualization

```bash
pip install matplotlib>=3.5.0
```

For plotting training results and network diagrams.

### Experiment Tracking

```bash
pip install wandb>=0.15.0
pip install tensorboard>=2.13.0
```

Track training runs with Weights & Biases or TensorBoard.

---

## Verify Installation

Test your installation with:

```python
# Test core environment
from powergrid.envs.multi_agent import MultiAgentMicrogrids

env = MultiAgentMicrogrids({'train': True, 'episode_length': 24})
obs, info = env.reset()
print("✓ PowerGrid 2.0 installed successfully!")
print(f"Agents: {env.agents}")
```

Expected output:
```
✓ PowerGrid 2.0 installed successfully!
Agents: ['MG1', 'MG2', 'MG3']
```

---

## Development Installation

If you plan to contribute or modify the code:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or install manually
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
pip install ruff>=0.1.0
pip install black>=22.0.0
```

Run tests to verify:

```bash
pytest tests/
```

---

## GPU Support (Optional)

For faster training with GPU:

```bash
# Install PyTorch with CUDA support
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

Configure RLlib to use GPU in your training script:

```python
config = PPOConfig().resources(num_gpus=1)
```

---

## Common Issues

### Issue: `pandapower` import error

**Solution**: Ensure `pandas>=2.3.1` is installed:
```bash
pip install --upgrade pandas>=2.3.1
```

### Issue: Ray RLlib version conflict

**Solution**: Ensure `numpy<2.0`:
```bash
pip install "numpy>=1.21.0,<2.0"
```

### Issue: Module not found after installation

**Solution**: Make sure you're in the activated virtual environment:
```bash
which python  # Should point to .venv/bin/python
```

---

## Next Steps

- Read [Basic Concepts](basic_concepts.md) to understand PowerGrid 2.0 architecture
- Follow the [Getting Started Guide](../getting_started.md) for your first multi-agent simulation
- Explore [Configuration Options](configuration.md) to customize your environment

---

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/your-lab/powergrid/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-lab/powergrid/discussions)
- **Email**: zwang@moveworks.ai
