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
| Configuration loading | [`textualrl/cli.py`](../textualrl/cli.py) accepts the flat public presets; [`textualrl/config.py`](../textualrl/config.py) also retains inherited structured-config support. |
| Rollout, critique, update, validation, and resume orchestration | [`textualrl/engine/trainer.py`](../textualrl/engine/trainer.py) |
| Same-task grouping, outcome routing, and critic calls | [`textualrl/gradient/reflect.py`](../textualrl/gradient/reflect.py), [`textualrl/optimizer/group_relative.py`](../textualrl/optimizer/group_relative.py) |
| Context-bounded analyst requests | [`textualrl/gradient/context_batching.py`](../textualrl/gradient/context_batching.py) |
| Evidence cards and cross-group proposal review | [`textualrl/optimizer/cross_group.py`](../textualrl/optimizer/cross_group.py), [`textualrl/gradient/aggregate.py`](../textualrl/gradient/aggregate.py) |
| Edit ranking, schedule, and patch application | [`textualrl/optimizer/clip.py`](../textualrl/optimizer/clip.py), [`scheduler.py`](../textualrl/optimizer/scheduler.py), [`skill.py`](../textualrl/optimizer/skill.py) |
| Runtime-observability filtering | [`textualrl/optimizer/quarantine.py`](../textualrl/optimizer/quarantine.py) |
| Validation acceptance and best tracking | [`textualrl/evaluation/gate.py`](../textualrl/evaluation/gate.py) |
| Inherited Meta and Slow mechanisms | [`textualrl/optimizer/meta_skill.py`](../textualrl/optimizer/meta_skill.py), [`slow_update.py`](../textualrl/optimizer/slow_update.py) |
| Benchmark loaders, execution, and metrics | [`textualrl/envs/`](../textualrl/envs/), [`textualrl/datasets/base.py`](../textualrl/datasets/base.py) |
| Shared optimizer prompts and benchmark-specific prompts | [`textualrl/prompts/`](../textualrl/prompts/), `textualrl/envs/<benchmark>/prompts/` |

## Inherited Meta and Slow settings

The provided configurations retain `use_meta_skill: true` and `use_slow_update: true`. Meta produces optimizer-side memory from longitudinal comparisons between contexts. Slow produces guidance inserted into the active context at epoch boundaries. These mechanisms can require extra target rollouts and optimizer calls beyond the ordinary step.

With `slow_update_gate_with_selection: false`, a produced Slow update is injected without the ordinary step's validation comparison. This is the inherited unconditional Slow path; it does not mean all updates are validation-accepted. The prior validation score can remain the recorded reference until a later evaluation. The stored best context remains separate. When final test evaluation is enabled, the trainer also attempts a final-context validation and promotes that context if it improves over the stored best. See [checkpoint selection](running.md#best-on-validation-and-final-contexts) before reporting results.
