"""Power grid environment setups.

This module provides predefined environment configurations and utilities
for loading them. Each setup is a directory containing:
- config.yml: Environment configuration (agent hierarchy, parameters)
- data.pkl: Time series data (load profiles, prices, etc.)

Available setups:
- ieee34_ieee13: IEEE 34-bus DSO with three IEEE 13-bus microgrids

Usage:
    from powergrid.setups import load_setup, get_available_setups

    # List available setups
    setups = get_available_setups()

    # Load a specific setup
    config = load_setup("ieee34_ieee13")
"""

from powergrid.setups.loader import load_setup, load_dataset, get_available_setups

__all__ = ["load_setup", "load_dataset", "get_available_setups"]
