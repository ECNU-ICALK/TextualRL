"""TextualRL Engine -- the training runner.

Analogous to the Runner in mmengine: orchestrates the full training pipeline
including rollout, gradient computation, aggregation, optimization, and
evaluation.
"""
from textualrl.engine.trainer import TextualRLTrainer  # noqa: F401

__all__ = ["TextualRLTrainer"]
