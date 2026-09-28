"""SkillOpt Optimizer -- skill update operations.

Analogous to the optimizer in neural network training: applies the computed
"gradient" (patches) to the current skill document to produce an updated
candidate skill.

Modules
-------
- skill: edit application (optimizer.step() / parameter update)
- clip: edit ranking and selection (gradient clipping)
- slow_update: longitudinal comparison and guidance (EMA / regularization)
- meta_skill: cross-epoch memory for optimizer context
"""
from textualrl.optimizer.skill import apply_edit, apply_patch  # noqa: F401
from textualrl.optimizer.clip import rank_and_select  # noqa: F401
