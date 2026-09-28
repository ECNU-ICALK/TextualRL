# Release provenance and reproducibility scope

This package was prepared from a research source snapshot on 2026-09-28. It derives from [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt), with TextualRL's groupwise critique and cross-group evidence changes in the retained `skillopt` runtime and a new `textualrl` launch interface.

The copied research launch configuration used `gpt-5.6-sol` for the optimizer. The public configuration examples instead expose the paper's default `gpt-5.5` optimizer and `Qwen3.8-27B` target through user-supplied endpoints. This is an explicit portability change. The release keeps inherited Meta and Slow settings and the unconditional Slow path described in [the method guide](method.md).

The release includes source, prompts, initial contexts, example settings, and data preparation instructions. It excludes training data, evaluation data, model weights, credentials, private endpoint settings, completed research runs, and trained contexts. No source revision from the upstream research run is asserted here, and the package version identifies this prepared release rather than an upstream experiment revision.

Packaging checks and dry-run configuration checks establish only that the distribution can be built and settings can be resolved. They do not establish benchmark performance. This release preparation has not rerun the paper's main experiments, and no numerical paper result is presented as a result from the portable examples.

## Recording a new experiment

Retain the resolved configuration and generated run directory for each target–benchmark pair. Record the endpoint's actual model/deployment version, dataset source and split preparation, initial skill, package revision, Python environment, token limits, temperatures, and seed settings. Seeds control sampling and request parameters where supported; an external service is not guaranteed to produce identical responses across requests or deployments.

Use validation for context selection and test data for evaluation of the selected context. Report `best_skill.md` and final-context scores under their distinct names. The inherited trainer can evaluate initial, best-on-validation, and final contexts in one completed run; these are different contexts, not interchangeable checkpoint labels. Do not choose a context by looking at its test score.

The exact dataset layout and runtime defaults are described in [data preparation](data.md) and [running experiments](running.md). Generated outputs contain model responses and task content. Keep them outside the source release unless you separately intend and are entitled to distribute that content.
