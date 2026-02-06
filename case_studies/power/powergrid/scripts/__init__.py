"""Production scripts for power grid training and evaluation.

Scripts:
    mappo_training.py: Production MAPPO/IPPO training script with:
        - W&B logging
        - Checkpointing and resumption
        - Event-driven validation
        - Command-line configuration

Usage:
    # Basic training
    python -m powergrid.scripts.mappo_training --iterations 100

    # With W&B logging
    python -m powergrid.scripts.mappo_training --iterations 100 --wandb

    # Quick test
    python -m powergrid.scripts.mappo_training --test
"""
