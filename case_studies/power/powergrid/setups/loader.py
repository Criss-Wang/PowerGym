"""Setup loading utilities for powergrid case study.

A setup is a complete environment definition containing:
- config.yml: Environment configuration (agent hierarchy, parameters)
- data.pkl: Time series data (load profiles, prices, etc.)

Each setup is a subdirectory under powergrid/setups/ containing these files.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Setups directory is this directory
SETUPS_DIR = Path(__file__).parent


def load_setup(setup_name: str) -> Dict[str, Any]:
    """Load a complete environment setup.

    Loads the config.yml from the setup directory and resolves the
    dataset_path to an absolute path.

    Args:
        setup_name: Name of the setup directory (e.g., 'ieee34_ieee13')

    Returns:
        Dictionary containing the environment configuration with resolved paths
    """
    setup_dir = SETUPS_DIR / setup_name

    if not setup_dir.exists():
        raise FileNotFoundError(f"Setup not found: {setup_dir}")

    # Load config
    config_path = setup_dir / "config.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve dataset_path to absolute path
    if "dataset_path" in config:
        data_path = setup_dir / config["dataset_path"]
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        config["dataset_path"] = str(data_path)

    return config


def get_available_setups() -> List[str]:
    """Get list of available setups.

    Returns:
        List of setup names (directory names containing config.yml)
    """
    return [
        d.name for d in SETUPS_DIR.iterdir()
        if d.is_dir() and (d / "config.yml").exists()
    ]


def load_dataset(file_path: str) -> Dict[str, Any]:
    """Load dataset from pickle file.

    Args:
        file_path: Absolute path to the dataset file (from load_setup)

    Returns:
        Loaded dataset (typically a dict with 'train' and 'test' keys)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)
