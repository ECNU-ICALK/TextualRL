# Method-to-code map

TextualRL optimizes a context document shared across tasks. The target model's parameters remain fixed. A task's reward is the benchmark's binary success signal; the optimizer produces text edits rather than parameter gradients.

## Groupwise Policy Critique

For each training task, the trainer gathers `same_task_rollouts` trajectories with their original rollout references. Outcome routing preserves task blocks. A mixed-outcome task is critiqued through within-task success/failure contrast. All-success and all-failure blocks are grouped across distinct tasks to find shared behaviors to preserve or failure mechanisms to repair. Homogeneous analysis requires evidence from at least two distinct tasks; repeated rollouts of one task do not become cross-task support.

The critic returns edit proposals and evidence cards independently. A card records observable conditions, actions, effects, and supporting references; cards can constrain other proposals even if their own group proposes no edit. Duplicate trajectory handling preserves outcome labels, references, and multiplicity.

## Cross-Group Policy Update

Hierarchical merging consolidates proposals. Cross-group review uses pooled current-step evidence to retain, narrow, or revise an edit's scope. Review does not require evidence from every outcome category and does not add target rollouts. The update is bounded by an edit budget and uses `append`, `insert_after`, `replace`, or `delete` operations. The existing runtime-observability filter excludes instructions that explicitly depend on information unavailable to the target agent.

The ordinary step evaluates the complete candidate context on held-out validation tasks. With the provided hard-score selection, a strict improvement updates the active context and recorded reference score; rejection preserves both. This score concerns the edit combination, so rejection alone does not establish that every individual edit is harmful. Within-epoch feedback records observed failures and rejected combinations for subsequent critique.

## Implementation map

| Responsibility | Main files |
| --- | --- |
| Portable train/eval CLI and endpoint roles | [`textualrl/cli.py`](../textualrl/cli.py), [`scripts/train.py`](../scripts/train.py), [`scripts/eval_only.py`](../scripts/eval_only.py) |
| Configuration loading | [`textualrl/cli.py`](../textualrl/cli.py) accepts the flat public presets; [`skillopt/config.py`](../skillopt/config.py) also retains inherited structured-config support. |
| Rollout, critique, update, validation, and resume orchestration | [`skillopt/engine/trainer.py`](../skillopt/engine/trainer.py) |
| Same-task grouping, outcome routing, and critic calls | [`skillopt/gradient/reflect.py`](../skillopt/gradient/reflect.py), [`skillopt/optimizer/group_relative.py`](../skillopt/optimizer/group_relative.py) |
| Context-bounded analyst requests | [`skillopt/gradient/context_batching.py`](../skillopt/gradient/context_batching.py) |
| Evidence cards and cross-group proposal review | [`skillopt/optimizer/cross_group.py`](../skillopt/optimizer/cross_group.py), [`skillopt/gradient/aggregate.py`](../skillopt/gradient/aggregate.py) |
| Edit ranking, schedule, and patch application | [`skillopt/optimizer/clip.py`](../skillopt/optimizer/clip.py), [`scheduler.py`](../skillopt/optimizer/scheduler.py), [`skill.py`](../skillopt/optimizer/skill.py) |
| Runtime-observability filtering | [`skillopt/optimizer/quarantine.py`](../skillopt/optimizer/quarantine.py) |
| Validation acceptance and best tracking | [`skillopt/evaluation/gate.py`](../skillopt/evaluation/gate.py) |
| Inherited Meta and Slow mechanisms | [`skillopt/optimizer/meta_skill.py`](../skillopt/optimizer/meta_skill.py), [`slow_update.py`](../skillopt/optimizer/slow_update.py) |
| Benchmark loaders, execution, and metrics | [`skillopt/envs/`](../skillopt/envs/), [`skillopt/datasets/base.py`](../skillopt/datasets/base.py) |
| Shared optimizer prompts and benchmark-specific prompts | [`skillopt/prompts/`](../skillopt/prompts/), `skillopt/envs/<benchmark>/prompts/` |

Some inherited names such as `ReflACTTrainer`, `gradient`, and `skillopt` remain in the source and artifacts. They are compatibility names rather than additional optimization methods introduced by this release. The source contains optional inherited research branches beyond the defaults; their presence does not imply that every branch is part of the paper configuration.

## Inherited Meta and Slow settings

The provided configurations retain `use_meta_skill: true` and `use_slow_update: true`. Meta produces optimizer-side memory from longitudinal comparisons between contexts. Slow produces guidance inserted into the active context at epoch boundaries. These mechanisms can require extra target rollouts and optimizer calls beyond the ordinary step.

With `slow_update_gate_with_selection: false`, a produced Slow update is injected without the ordinary step's validation comparison. This is the inherited unconditional Slow path; it does not mean all updates are validation-accepted. The prior validation score can remain the recorded reference until a later evaluation. The stored best context remains separate. When final test evaluation is enabled, the trainer also attempts a final-context validation and promotes that context if it improves over the stored best. See [checkpoint selection](running.md#best-on-validation-and-final-contexts) before reporting results.
