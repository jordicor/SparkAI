"""
SparkAI Database Seed Module

Run the seed with:
    python -m data.seed.seed_data

Or import and use programmatically:
    from data.seed.seed_data import run_seed
    run_seed()
"""

from .seed_data import run_seed

__all__ = ["run_seed"]
