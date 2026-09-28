"""Role-isolated settings, endpoint quotas, and content-free request evidence.

This module deliberately does not import the model backend: its environment must
be configured before the original backend creates its module-level settings.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit


def endpoint_urls(base_url: str, api_style: str) -> list[str]:
    suffix = "/responses" if api_style == "responses" else "/chat/completions"
    urls = []
    for value in str(base_url).split(","):
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Base URLs must be HTTP(S) URLs without credentials, queries, or fragments")
        if value.endswith(("/responses", "/chat/completions")) and not value.endswith(suffix):
            raise ValueError("Endpoint suffix does not match its configured API style")
        urls.append(value if value.endswith(suffix) else value + suffix)
    if len(urls) != len(set(urls)):
        raise ValueError("Endpoint pools must contain distinct URLs")
    return urls


def endpoint_limits(config: dict) -> dict[str, int]:
    """Share capacity across roles using the same request endpoint."""
    limits: dict[str, int] = {}
    for role in ("target", "optimizer"):
        urls = endpoint_urls(config[f"{role}_qwen_chat_base_url"], config[f"{role}_api_style"])
        value = config[f"{role}_endpoint_concurrency"]
        capacities = value if isinstance(value, list) else [value] * len(urls)
        if len(capacities) != len(urls):
            raise ValueError(f"{role}_endpoint_concurrency requires one quota per endpoint")
        for url, capacity in zip(urls, capacities):
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                raise ValueError("Endpoint concurrency must be a positive integer")
            # A shared endpoint never exceeds either role's declared quota.
            limits[url] = min(limits.get(url, capacity), capacity)
    return limits


def configure_environment(config: dict, action: str) -> None:
    """Only the two public credential variables may supply authentication."""
    required = ("TARGET_API_KEY", "OPTIMIZER_API_KEY") if action == "train" else ("TARGET_API_KEY",)
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError("Missing required environment variable(s): " + ", ".join(missing))
    # Empty role settings would fall back to generic Qwen settings in the core.
    # Clear that fallback, and replace every role-specific transport setting.
    for key in list(os.environ):
        if key.startswith(("QWEN_CHAT_", "TARGET_QWEN_CHAT_", "OPTIMIZER_QWEN_CHAT_")):
            del os.environ[key]
    for role in ("target", "optimizer"):
        prefix = role.upper() + "_QWEN_CHAT_"
        values = {
            "API_KEY": os.environ.get(role.upper() + "_API_KEY", "").strip(),
            "BASE_URL": config[f"{role}_qwen_chat_base_url"],
            "MODEL": config[f"{role}_model"],
            "THINKING_API": config[f"{role}_thinking_api"],
            "REASONING_EFFORT": config[f"{role}_reasoning_effort"],
            "API_STYLE": config[f"{role}_api_style"],
            "TEMPERATURE": config[f"{role}_qwen_chat_temperature"],
            "ENABLE_THINKING": config[f"{role}_qwen_chat_enable_thinking"],
            "OMIT_OUTPUT_LIMIT": config[f"{role}_omit_output_limit"],
            "USE_MAX_COMPLETION_TOKENS": config.get(f"{role}_qwen_chat_use_max_completion_tokens", False),
            # The core performs integer arithmetic even when the wire limit is
            # omitted. A null preset cap means "omit", not an env string "none".
            "MAX_TOKENS": config.get(f"{role}_qwen_chat_max_tokens") or 32768,
            "TIMEOUT_SECONDS": config.get(f"{role}_qwen_chat_timeout_seconds") or 3600,
            "FORCE_DO_SAMPLE": config.get(f"{role}_force_do_sample", False),
        }
        for key, value in values.items():
            os.environ[prefix + key] = ("none" if value is None else
                                       str(value).lower() if isinstance(value, bool) else str(value))
    if config.get("alfworld_data_root"):
        os.environ["ALFWORLD_DATA"] = str(config["alfworld_data_root"])
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")


def _numeric_usage(value):
    if not isinstance(value, dict):
        return None
    return {key: (_numeric_usage(item) if isinstance(item, dict) else item)
            for key, item in value.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
                       "output_tokens", "input_tokens_details", "output_tokens_details",
                       "prompt_tokens_details", "completion_tokens_details", "cached_tokens",
                       "reasoning_tokens", "audio_tokens", "accepted_prediction_tokens",
                       "rejected_prediction_tokens"}
            and (isinstance(item, (int, float, dict)) and not isinstance(item, bool))}


class RequestFailure(RuntimeError):
    """Safe transport error retaining status and the core's control signals."""

    def __init__(self, role: str, error_type: str, status: int | None, reason: str | None):
        self.code = self.status_code = status
        self.error_type = error_type
        self.reason = reason
        detail = f", HTTP {status}" if status is not None else ""
        marker = f", {reason}" if reason else ""
        super().__init__(f"{role} request failed ({error_type}{detail}{marker}); see runtime_requests.jsonl")


def _safe_failure(role: str, error: Exception) -> RequestFailure:
    """Extract allowlisted diagnostics without retaining an echoed request body.

    The core context splitter inspects exception text for HTTP 400 together
    with an explicit overflow marker. Removing either silently disables its
    sibling-preserving split/retry behavior.
    """
    status = None
    chain = []
    current = error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        code = getattr(current, "code", None) or getattr(current, "status_code", None)
        if status is None and isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
            status = code
        current = current.__cause__ or current.__context__
    # Inspect locally; no raw message is copied into the sanitized exception
    # message or included in persisted request/split evidence.
    message = str(error).lower()
    if status is None:
        match = re.search(r"\bhttp\s+([45]\d\d)\b", message)
        status = int(match.group(1)) if match else None
    reason = None
    if status == 400 and any(term in message for term in (
        "maximum context length", "context_length_exceeded", "exceeds the context window",
    )):
        reason = "context_length_exceeded"
    elif status == 429:
        reason = "rate_limit_exceeded"
    elif status in {408, 504} or any(isinstance(item, TimeoutError) for item in chain):
        reason = "timeout"
    elif status == 401:
        reason = "authentication_failed"
    elif status == 403:
        reason = "permission_denied"
    return RequestFailure(role, type(error).__name__, status, reason)


@contextmanager
def request_runtime(backend, config: dict, audit_path: Path):
    """Bound actual in-flight HTTP calls; audit metadata without prompts or keys."""
    limits = endpoint_limits(config)
    slots = {url: threading.BoundedSemaphore(limit) for url, limit in limits.items()}
    counts = {url: {"active": 0, "peak": 0, "calls": 0, "errors": 0} for url in limits}
    lock = threading.Lock()
    original = backend._post_chat_completion
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    def post(payload, timeout, request_config):
        if request_config is backend.TARGET_CONFIG:
            role = "target"
        elif request_config is backend.OPTIMIZER_CONFIG:
            role = "optimizer"
        else:
            raise ValueError("Request does not belong to a configured target or optimizer role")
        url = backend._chat_url(request_config)
        if url not in slots:
            raise ValueError("Request endpoint has no configured concurrency quota")
        with slots[url]:
            before = time.monotonic()
            with lock:
                count = counts[url]
                count["active"] += 1
                count["peak"] = max(count["peak"], count["active"])
            row = {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": url, "role": role, "api_style": request_config.api_style,
                "model": config[f"{role}_model"], "timeout": timeout or request_config.timeout_seconds,
            }
            # These are generated controls, never conversation text or headers.
            for key in ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "seed"):
                value = payload.get(key)
                row[key] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            template = payload.get("chat_template_kwargs") or {}
            if isinstance(template.get("enable_thinking"), bool):
                row["chat_template_kwargs"] = {"enable_thinking": template["enable_thinking"]}
            thinking = payload.get("thinking") or {}
            if thinking.get("type") in {"enabled", "disabled"}:
                row["thinking"] = {"type": thinking["type"]}
            effort = payload.get("reasoning_effort") or (payload.get("reasoning") or {}).get("effort")
            if effort in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                row["reasoning_effort"] = effort
            try:
                result = original(payload, timeout, replace(request_config, base_url=url))
                chat_result = (backend._chat_completion_from_response(result)
                               if request_config.api_style == "responses" else result)
                choice = (chat_result.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                row.update(usage=_numeric_usage(result.get("usage")),
                           content_chars=len(content), reasoning_chars=len(reasoning),
                           tool_calls=len(message.get("tool_calls") or []),
                           has_output=bool(str(content).strip() or message.get("tool_calls")))
                if choice.get("finish_reason") in {"stop", "length", "tool_calls", "content_filter", "function_call"}:
                    row["finish_reason"] = choice["finish_reason"]
                if request_config.api_style == "responses":
                    row["instructions_chars"] = len(payload.get("instructions") or "")
                    row["response_instructions_match"] = (
                        result["instructions"] == payload.get("instructions")
                        if result.get("instructions") is not None else None)
                return result
            except Exception as error:
                safe_error = _safe_failure(role, error)
                row.update(request_failed=True, error_type=type(error).__name__,
                           http_status=safe_error.code, error_reason=safe_error.reason)
                with lock:
                    count["errors"] += 1
                # Gateways sometimes echo request headers or content in errors.
                # Keep status/type evidence without propagating that body into
                # the core's retry messages and trajectory logs.
                raise safe_error from None
            finally:
                row["finished_at"] = datetime.now(timezone.utc).isoformat()
                row["seconds"] = round(time.monotonic() - before, 3)
                with lock:
                    count["active"] -= 1
                    count["calls"] += 1
                    with audit_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    backend._post_chat_completion = post
    try:
        yield counts
    finally:
        backend._post_chat_completion = original
