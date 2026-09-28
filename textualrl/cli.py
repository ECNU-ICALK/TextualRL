"""Resolve portable public presets and dispatch to the original engine scripts."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import runpy
import sys

import yaml

from textualrl.runtime import configure_environment, endpoint_limits, request_runtime

REPO_ROOT = Path(__file__).resolve().parents[1]
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ALLOWED_VARIABLES = {
    "TEXTUALRL_DATA_DIR", "TEXTUALRL_OUTPUT_DIR", "TEXTUALRL_TARGET_MODEL",
    "TEXTUALRL_OPTIMIZER_MODEL", "TARGET_BASE_URL", "OPTIMIZER_BASE_URL", "TEXTUALRL_REPO_ROOT",
}
_SECRET_FIELD = re.compile(r"(?:^|_)(?:api_key|password|secret|authorization|access_token)(?:$|_)", re.I)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train or evaluate TextualRL skills with portable presets.")
    subcommands = result.add_subparsers(dest="action", required=True)
    for action in ("train", "eval"):
        sub = subcommands.add_parser(action)
        sub.add_argument("--config", type=Path, required=True, help="Flat YAML benchmark preset")
        sub.add_argument("--dry-run", action="store_true", help="Print resolved settings without credentials, data loading, or API calls")
        sub.add_argument("--data-dir", type=Path, help="Root substituted for TEXTUALRL_DATA_DIR")
        sub.add_argument("--output-dir", type=Path, help="Exact output directory (overrides out_root)")
        sub.add_argument("--target-model", help="Target deployment identifier")
        sub.add_argument("--optimizer-model", help="Optimizer deployment identifier")
        if action == "train":
            sub.add_argument("--resume-from", type=Path, help="Existing run directory; reuse the trainer's runtime state/history")
        else:
            sub.add_argument("--skill", type=Path, help="Skill markdown to evaluate; defaults to skill_init")
            sub.add_argument("--split", default="test", choices=("train", "val", "test", "valid_seen", "valid_unseen", "all"))
    return result


def _substitute(value, variables: dict):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            if _SECRET_FIELD.search(key):
                raise ValueError("Credentials must only be supplied through TARGET_API_KEY and OPTIMIZER_API_KEY")
            result[key] = _substitute(item, variables)
        return result
    if isinstance(value, list):
        return [_substitute(item, variables) for item in value]
    if isinstance(value, str):
        def replace_variable(match):
            name = match.group(1)
            if name not in _ALLOWED_VARIABLES:
                raise ValueError(f"Unsupported configuration variable: {name}")
            if not str(variables.get(name, "")).strip():
                raise ValueError(f"Missing configuration environment variable: {name}")
            return str(variables[name])
        result = _VARIABLE.sub(replace_variable, value)
        if "${" in result:
            raise ValueError("Malformed environment variable reference in configuration")
        return result
    return value


def resolve_config(args: argparse.Namespace, environ=None) -> dict:
    environ = os.environ if environ is None else environ
    with args.config.expanduser().open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Configuration must be a nonempty flat YAML mapping")
    if "_base_" in raw or any(isinstance(raw.get(key), dict) for key in ("model", "train", "gradient", "optimizer", "evaluation", "env")):
        raise ValueError("The public CLI accepts flat, self-contained YAML presets")
    defaults = {
        "TEXTUALRL_REPO_ROOT": str(REPO_ROOT),
        "TEXTUALRL_DATA_DIR": str(REPO_ROOT / "data"),
        "TEXTUALRL_OUTPUT_DIR": str(REPO_ROOT / "outputs"),
        "TEXTUALRL_TARGET_MODEL": "Qwen3.8-27B",
        "TEXTUALRL_OPTIMIZER_MODEL": "gpt-5.5",
        "TARGET_BASE_URL": "http://localhost:8000/v1",
        "OPTIMIZER_BASE_URL": "https://api.openai.com/v1",
    }
    variables = {key: environ.get(key, value) for key, value in defaults.items()}
    variables["TEXTUALRL_REPO_ROOT"] = str(REPO_ROOT)
    if args.data_dir is not None:
        variables["TEXTUALRL_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    if args.output_dir is not None:
        variables["TEXTUALRL_OUTPUT_DIR"] = str(args.output_dir.expanduser().resolve())
    if args.target_model is not None:
        variables["TEXTUALRL_TARGET_MODEL"] = args.target_model
    if args.optimizer_model is not None:
        variables["TEXTUALRL_OPTIMIZER_MODEL"] = args.optimizer_model
    for key in ("TEXTUALRL_DATA_DIR", "TEXTUALRL_OUTPUT_DIR"):
        if variables[key]:
            variables[key] = str(Path(variables[key]).expanduser().resolve())
    cfg = _substitute(raw, variables)
    if not isinstance(cfg.get("env"), str) or not cfg["env"].strip():
        raise ValueError("Configuration requires a benchmark env name")
    for role in ("target", "optimizer"):
        cfg.setdefault(f"{role}_backend", "qwen_chat")
        if cfg[f"{role}_backend"] not in {"qwen", "qwen_chat"}:
            raise ValueError("The public runtime uses the qwen_chat OpenAI-compatible transport for both roles")
        cfg[f"{role}_backend"] = "qwen_chat"
        override = getattr(args, role + "_model")
        cfg[f"{role}_model"] = override or cfg.get(f"{role}_model") or variables[f"TEXTUALRL_{role.upper()}_MODEL"]
        cfg.setdefault(f"{role}_qwen_chat_base_url", variables[role.upper() + "_BASE_URL"])
        cfg.setdefault(f"{role}_thinking_api", "chat_template" if role == "target" else "reasoning_effort")
        cfg.setdefault(f"{role}_reasoning_effort", "none" if role == "target" else "medium")
        cfg.setdefault(f"{role}_api_style", "chat_completions" if role == "target" else "responses")
        cfg.setdefault(f"{role}_endpoint_concurrency", 8)
        cfg.setdefault(f"{role}_qwen_chat_temperature", 0.7 if role == "target" else None)
        cfg.setdefault(f"{role}_qwen_chat_enable_thinking", role == "optimizer" and cfg[f"{role}_reasoning_effort"] != "none")
        cfg.setdefault(f"{role}_omit_output_limit", cfg.get(f"{role}_qwen_chat_omit_output_limit", role == "optimizer"))
        if cfg[f"{role}_thinking_api"] not in {"chat_template", "reasoning_effort", "provider", "omit", "default", "auto"}:
            raise ValueError(f"Invalid {role}_thinking_api")
        if cfg[f"{role}_reasoning_effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Invalid {role}_reasoning_effort")
        if cfg[f"{role}_api_style"] not in {"chat_completions", "responses"}:
            raise ValueError(f"Invalid {role}_api_style")
        if cfg[f"{role}_api_style"] == "responses" and cfg[f"{role}_thinking_api"] not in {"reasoning_effort", "omit"}:
            raise ValueError("Responses transport requires thinking_api=reasoning_effort or omit")
        for suffix in ("qwen_chat_enable_thinking", "omit_output_limit"):
            if not isinstance(cfg[f"{role}_{suffix}"], bool):
                raise ValueError(f"{role}_{suffix} must be a YAML boolean")
    cfg.setdefault("seed", 42)
    cfg.setdefault("evaluation_target_temperature", 0.0)
    cfg.setdefault("evaluation_target_seed", 42)
    cfg.setdefault("out_root", str(Path(variables["TEXTUALRL_OUTPUT_DIR"]) / cfg["env"]))
    if args.output_dir is not None:
        cfg["out_root"] = str(args.output_dir.expanduser().resolve())
    elif args.action == "eval":
        cfg["out_root"] = str(Path(cfg["out_root"]) / "eval")
    resume = getattr(args, "resume_from", None)
    if resume is not None:
        resume = resume.expanduser().resolve()
        if args.output_dir is not None and resume != args.output_dir.expanduser().resolve():
            raise ValueError("--resume-from and --output-dir must refer to the same run directory")
        if not resume.is_dir() or not any((resume / name).is_file() for name in ("runtime_state.json", "history.json")):
            raise ValueError("--resume-from requires a run directory containing runtime_state.json or history.json")
        cfg["out_root"] = str(resume)
    cfg["out_root"] = str(Path(cfg["out_root"]).expanduser().resolve())
    if cfg.get("skill_init"):
        cfg["skill_init"] = str(Path(cfg["skill_init"]).expanduser().resolve())
    if args.action == "eval":
        skill = args.skill or cfg.get("skill_init")
        if not skill:
            raise ValueError("Evaluation requires --skill or skill_init in the config")
        args.skill = Path(skill).expanduser().resolve()
        if args.split == "all" and cfg["env"] == "alfworld":
            raise ValueError("Evaluate ALFWorld one split at a time: --split train, val, or test")
    endpoint_limits(cfg)
    return cfg


def run(args: argparse.Namespace, cfg: dict) -> None:
    configure_environment(cfg, args.action)
    # Import only after role-specific credentials and transport controls exist.
    from textualrl import model
    from textualrl.model import qwen_backend
    # Also isolate repeated in-process invocations from previously imported state.
    qwen_backend.TARGET_CONFIG = qwen_backend._initial_config("target")
    qwen_backend.OPTIMIZER_CONFIG = qwen_backend._initial_config("optimizer")
    model.set_target_backend("qwen_chat")
    model.set_optimizer_backend("qwen_chat")
    output = Path(cfg["out_root"])
    output.mkdir(parents=True, exist_ok=True)
    effective = output / "effective_config.json"
    effective.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    script = REPO_ROOT / "scripts" / ("train.py" if args.action == "train" else "eval_only.py")
    argv = [str(script), "--config", str(effective),
            "--target_model", cfg["target_model"], "--optimizer_model", cfg["optimizer_model"]]
    if args.action == "eval":
        argv.extend(["--skill", str(args.skill), "--split", args.split])
    previous_argv = sys.argv
    overrides = (model.target_generation_overrides(
        temperature=cfg["evaluation_target_temperature"], seed=cfg["evaluation_target_seed"])
        if args.action == "eval" else nullcontext())
    try:
        sys.argv = argv
        with request_runtime(qwen_backend, cfg, output / "runtime_requests.jsonl"), overrides:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv


def main(argv=None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        cfg = resolve_config(args)
        if args.dry_run:
            result = {"action": args.action, "dry_run": True, "config": cfg,
                      "endpoint_limits": endpoint_limits(cfg),
                      "required_credentials": ["TARGET_API_KEY"] + (["OPTIMIZER_API_KEY"] if args.action == "train" else [])}
            if args.action == "eval":
                result.update(skill=str(args.skill), split=args.split)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            run(args, cfg)
    except yaml.YAMLError:
        # YAML parser diagnostics may quote a line containing a pasted secret.
        argument_parser.error("Invalid YAML configuration")
    except (ValueError, OSError) as error:
        argument_parser.error(str(error))
    return 0
