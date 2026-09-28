# Third-party notices

## Microsoft SkillOpt

This release is derived from [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt). Its training loop, benchmark adapters, model clients, optimizer utilities, prompts, and package organization form the foundation of the `textualrl` runtime.

The upstream MIT license is reproduced verbatim in [LICENSE](LICENSE), including:

> Copyright (c) 2026 Microsoft Corporation

TextualRL adds groupwise critique and cross-group evidence behavior, portable configuration and launch support, and release documentation. The implementation uses the `textualrl` namespace. This package is a modified derivative; it is not a Microsoft product release. No upstream Git revision is asserted for the supplied research snapshot.

## Vendored ALFWorld helpers from SkillRL / verl-agent

`textualrl/envs/alfworld/vendor/` contains modified helpers inherited through SkillOpt. Their source headers identify SkillRL's `agent_system` package and the Apache License, Version 2.0. The historical `NTU-LANTERN/SkillRL` URL in those headers was unavailable when this release was prepared. The matching source paths are publicly available in [aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL/tree/main/agent_system).

| Bundled file | Source path named by the inherited header |
| --- | --- |
| `alfworld_envs.py` | `agent_system/environments/env_package/alfworld/envs.py` |
| `alfworld_projection.py` | `agent_system/environments/env_package/alfworld/projection.py` |
| `env_base.py` | `agent_system/environments/base.py` |
| `env_manager.py` | `agent_system/environments/env_manager.py` |
| `alfworld_prompts.py` | `agent_system/environments/prompts/alfworld.py` |
| `memory.py` | `agent_system/memory/base.py` and `agent_system/memory/memory.py` |

These files are adapted subsets, not byte-identical copies of the current source. Local changes include package import paths, the text-only environment wrapper, process-based environment workers, prompt loading, and optional Torch handling. The exact source revision of the inherited subset is not recorded in this release.

The corresponding environment source files retain the following notice:

> Copyright 2025 Nanyang Technological University (NTU), Singapore
> and the verl-agent (GiGPO) team.

The public SkillRL repository also provides an MIT project license and a notice crediting inherited verl code. Copies are included here to retain those attributions alongside the file-level Apache notice:

- [Apache License, Version 2.0](docs/licenses/Apache-2.0.txt), from the [Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt).
- [SkillRL MIT license](docs/licenses/SkillRL-MIT.txt), from [SkillRL/LICENSE](https://github.com/aiming-lab/SkillRL/blob/main/LICENSE).
- [SkillRL notice](docs/licenses/SkillRL-NOTICE.txt), from [SkillRL/Notice.txt](https://github.com/aiming-lab/SkillRL/blob/main/Notice.txt).
- [Example file-level Apache notice](https://github.com/aiming-lab/SkillRL/blob/main/agent_system/environments/env_package/alfworld/envs.py).

The ALFWorld runtime package is installed separately; its license and the terms for its environment assets continue to apply.

## Datasets, model services, and other dependencies

No benchmark examples, document corpus, spreadsheets, document images, or model weights are distributed in this source package. Identifier/path manifests and preparation scripts, where included, describe separately obtained benchmark sources. Consult [the data guide](docs/data.md) and each provider's own terms before redistributing data or derived task content.

Python packages declared in `pyproject.toml` and external model services are separate dependencies. Their inclusion in an installation or a configuration does not change their licenses or grant access to hosted models.
