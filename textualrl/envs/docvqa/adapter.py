from __future__ import annotations

from textualrl.datasets.base import BatchSpec
from textualrl.envs.base import EnvAdapter
from textualrl.envs.docvqa.dataloader import DocVQADataLoader
from textualrl.envs.docvqa.rollout import run_batch
from textualrl.model import target_generation_overrides


class DocVQAAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 120,
        workers: int = 16,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        image_detail: str = "auto",
        max_completion_tokens: int = 16384,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.image_detail = image_detail
        self.dataloader = DocVQADataLoader(
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
        """Repeat each DocVQA item with collision-free execution IDs."""
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
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            image_detail=self.image_detail,
            max_completion_tokens=self.max_completion_tokens,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            task_timeout=self.exec_timeout,
            target_temperature=target_temperature,
            )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get("task_type") or "docvqa")
            if task_type not in seen:
                seen.append(task_type)
        return seen or ["docvqa"]
