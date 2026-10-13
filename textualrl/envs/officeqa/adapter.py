from __future__ import annotations

import os

from textualrl.datasets.base import BatchSpec
from textualrl.envs.base import EnvAdapter
from textualrl.envs.officeqa.dataloader import OfficeQADataLoader
from textualrl.envs.officeqa.rollout import run_batch
from textualrl.model import target_generation_overrides


class OfficeQAAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 8,
        analyst_workers: int = 8,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_tool_turns: int = 12,
        max_completion_tokens: int = 16384,
        search_mode: str = "offline",
        max_queries_per_turn: int = 4,
        search_api_url: str = os.environ.get("OFFICEQA_SEARCH_API_URL", "http://localhost:8080/search_tool/search"),
        search_auth_env: str = "OFFICEQA_CUSTOM_SEARCH_AUTH",
        search_provider: str = "duckduckgo",
        search_max_num_results: int = 4,
        search_timeout_seconds: int = 20,
        use_local_tools: bool = True,
        data_dirs: list[str] | str | None = None,
        docs_dirs: list[str] | str | None = None,    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_tool_turns = max_tool_turns
        self.max_completion_tokens = int(max_completion_tokens)
        self.search_mode = str(search_mode or "offline")
        self.max_queries_per_turn = int(max_queries_per_turn)
        self.search_api_url = str(search_api_url or "").strip()
        self.search_auth_env = str(search_auth_env or "OFFICEQA_CUSTOM_SEARCH_AUTH").strip()
        self.search_provider = str(search_provider or "duckduckgo").strip()
        self.search_max_num_results = int(search_max_num_results)
        self.search_timeout_seconds = int(search_timeout_seconds)
        self.use_local_tools = bool(use_local_tools)
        self.data_dirs = data_dirs if data_dirs is not None else docs_dirs
        self.dataloader = OfficeQADataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self.same_task_rollout_temperature = cfg.get(
            "same_task_rollout_temperature"
        )
        self.evaluation_target_temperature = cfg.get(
            "evaluation_target_temperature"
        )
        self.evaluation_target_seed = cfg.get(
            "evaluation_target_seed"
        )

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def expand_same_task_rollouts(
        self,
        env_manager,
        rollout_count: int,
    ) -> list[dict]:
        """Repeat each OfficeQA item with collision-free execution IDs."""
        rollout_count = int(rollout_count)
        if rollout_count <= 0:
            raise ValueError(
                f"rollout_count must be positive, got {rollout_count}"
            )
        items = list(env_manager)
        if rollout_count == 1:
            return items

        expanded: list[dict] = []
        for item in items:
            source_id = str(item.get("rollout_group_id") or item["id"])
            for rollout_index in range(1, rollout_count + 1):
                sample = dict(item)
                sample["rollout_group_id"] = source_id
                sample["rollout_index"] = rollout_index
                sample["rollout_count"] = rollout_count
                sample["id"] = (
                    f"{source_id}__sample_{rollout_index:02d}"
                    f"_of_{rollout_count:02d}"
                )
                expanded.append(sample)
        return expanded

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        items: list[dict] = env_manager
        grouped_training = any(
            int(item.get("rollout_count", 1) or 1) > 1
            for item in items
        )
        training_rollout = (
            bool(kwargs.get("use_eval_feedback", False)) or grouped_training
        )
        if "target_temperature" in kwargs:
            target_temperature = kwargs["target_temperature"]
        elif training_rollout:
            target_temperature = self.same_task_rollout_temperature
        else:
            target_temperature = self.evaluation_target_temperature
        if "target_seed" in kwargs:
            target_seed = kwargs["target_seed"]
        else:
            target_seed = (
                None
                if training_rollout
                else getattr(self, "evaluation_target_seed", None)
            )
        with target_generation_overrides(
            temperature=target_temperature,
            seed=target_seed,
        ):
            return run_batch(
                items=items,
            out_root=out_dir,
            skill_content=skill_content,
            workers=self.workers,
            max_tool_turns=self.max_tool_turns,
            max_completion_tokens=self.max_completion_tokens,
            search_mode=self.search_mode,
            max_queries_per_turn=self.max_queries_per_turn,
            search_api_url=self.search_api_url,
            search_auth_env=self.search_auth_env,
            search_provider=self.search_provider,
            search_max_num_results=self.search_max_num_results,
            search_timeout_seconds=self.search_timeout_seconds,
            use_local_tools=self.use_local_tools,
            data_dirs=self.data_dirs,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            target_temperature=target_temperature,
            )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get("task_type") or "officeqa")
            if task_type not in seen:
                seen.append(task_type)
        return seen or ["officeqa"]
