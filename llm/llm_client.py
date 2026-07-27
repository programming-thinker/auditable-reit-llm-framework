"""DeepSeek / Qwen API wrapper with disk caching and retry logic.

Key design points (CLAUDE.md Section 9):
- Responses cached via diskcache, key = sha256(prompt + model + decoding_params)
- Determinism: temperature=0, model version pinned in config/config.yaml
- Retry via tenacity with exponential backoff
- Logging via structlog to JSONL
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import structlog
import yaml
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ── Exceptions ────────────────────────────────────────────────────────────

_TRANSIENT_MARKERS = frozenset(
    [
        "429",
        "rate limit",
        "rate_limit",
        "timeout",
        "timed out",
        "connection",
        "overload",
        "overloaded",
        "temporarily",
        "server error",
        "service unavailable",
        "503",
        "502",
        "504",
    ]
)


class TransientAPIError(Exception):
    """Retryable API error (rate-limit, timeout, 5xx)."""


class PermanentAPIError(Exception):
    """Non-retryable API error (auth, invalid request)."""


def _classify_api_error(exc: Exception) -> TransientAPIError | PermanentAPIError:
    """Wrap a raw exception into a transient or permanent error."""
    text = str(exc).lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return TransientAPIError(str(exc))
    return PermanentAPIError(str(exc))


# ── Cache key ─────────────────────────────────────────────────────────────


def _cache_key(
    messages: list[dict[str, str]],
    model: str,
    decoding_params: dict[str, Any],
) -> str:
    """Compute a deterministic cache key.

    Key = sha256(canonicalized JSON of messages + model + params).
    """
    payload = json.dumps(
        {"messages": messages, "model": model, "params": decoding_params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Client ────────────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around DeepSeek / Qwen-compatible OpenAI-style API.

    Features:
    - Automatic disk caching (no duplicate API spend).
    - Exponential-backoff retry on transient errors.
    - Support for both thinking (streaming) and non-thinking modes.
    - Structured logging via structlog.

    Parameters
    ----------
    model_config_name : str
        Key in config/llm_model_configs.yaml (e.g. "qwen3_8b_non_thinking").
    config_path : Path
        Path to the LLM model config YAML.
    cache_dir : Path | None
        Directory for diskcache.  Defaults to .cache/llm_responses.
    """

    def __init__(
        self,
        model_config_name: str = "qwen3_8b_non_thinking",
        config_path: Path = Path("config/llm_model_configs.yaml"),
        cache_dir: Path | None = None,
    ) -> None:
        # ── load model config ─────────────────────────────────────────
        with config_path.open("r", encoding="utf-8") as f:
            all_configs = yaml.safe_load(f)
        if model_config_name not in all_configs:
            raise ValueError(
                f"Model config '{model_config_name}' not found in {config_path}. "
                f"Available: {list(all_configs)}"
            )
        self._config: dict[str, Any] = all_configs[model_config_name]
        self._config_name = model_config_name
        self._model: str = self._config["model"]
        self._enable_thinking: bool = bool(self._config.get("enable_thinking", False))

        # ── thinking-mode guard ───────────────────────────────────────
        # The thinking (streaming) path is NOT audit-safe as implemented:
        #   1. It sends no `temperature`, so the provider default applies and
        #      determinism (CLAUDE.md Section 9: temperature=0, pinned model)
        #      is silently violated.
        #   2. Streaming returns no usage object, so prompt/completion token
        #      counts are logged as 0 and the cost ledger under-reports spend.
        # No reported run ever used this path (all logged API calls show
        # enable_thinking=False). Fail fast unless a config entry explicitly
        # opts in with `allow_thinking_mode: true` after fixing both issues.
        self._allow_thinking_mode: bool = bool(
            self._config.get("allow_thinking_mode", False)
        )
        if self._enable_thinking and not self._allow_thinking_mode:
            raise NotImplementedError(
                f"Model config '{model_config_name}' sets enable_thinking=true, "
                "but the thinking (streaming) path is disabled: it does not "
                "send temperature=0 (non-deterministic) and returns empty "
                "token usage (breaks the audit_log cost ledger). Set "
                "'allow_thinking_mode: true' in the model config entry ONLY "
                "after fixing determinism and usage accounting in "
                "LLMClient._call_thinking_stream."
            )

        # ── OpenAI client ─────────────────────────────────────────────
        load_dotenv()
        api_key_env = self._config.get("api_key_env", "DASHSCOPE_API_KEY")
        base_url_env = self._config.get("base_url_env", "DASHSCOPE_BASE_URL")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key: set the {api_key_env} environment variable."
            )
        base_url = os.environ.get(
            base_url_env,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._api_key = api_key  # kept only for log redaction

        # ── disk cache ────────────────────────────────────────────────
        if cache_dir is None:
            cache_dir = Path(".cache/llm_responses")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(cache_dir))

        logger.info(
            "llm_client_init",
            model_config=model_config_name,
            model=self._model,
            enable_thinking=self._enable_thinking,
            cache_dir=str(cache_dir),
        )

    # ── public API ────────────────────────────────────────────────────

    def query(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        """Send a single user prompt (with optional system prompt).

        This is a convenience wrapper around :meth:`query_messages`.

        Returns
        -------
        dict with keys: content, usage, model, cached, latency_sec
        """
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.query_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def query_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        """Send a messages list and return the parsed response.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style messages list, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
        temperature : float
            Sampling temperature.  Default 0.0 for determinism.
        max_tokens : int
            Maximum completion tokens.
        json_mode : bool
            If True (default, non-thinking only), request JSON output format.

        Returns
        -------
        dict with keys:
            content (str) — model response text
            usage (dict)  — {prompt_tokens, completion_tokens, total_tokens}
            model (str)   — model identifier returned by API
            cached (bool) — True if served from disk cache
            latency_sec (float) — wall-clock seconds (0.0 if cached)
            has_reasoning_content (bool) — True if thinking tokens were present
        """
        decoding_params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "enable_thinking": self._enable_thinking,
        }

        key = _cache_key(messages, self._model, decoding_params)

        # ── cache hit ─────────────────────────────────────────────────
        cached_result = self._cache.get(key)
        if cached_result is not None:
            logger.debug("cache_hit", cache_key=key[:16])
            cached_result["cached"] = True
            cached_result["latency_sec"] = 0.0
            return cached_result

        # ── cache miss → call API ─────────────────────────────────────
        logger.info(
            "api_call_start",
            model=self._model,
            enable_thinking=self._enable_thinking,
            n_messages=len(messages),
        )

        t0 = time.monotonic()
        result = self._call_with_retry(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        latency = time.monotonic() - t0

        result["cached"] = False
        result["latency_sec"] = round(latency, 3)

        # ── store in cache ────────────────────────────────────────────
        self._cache.set(key, result)
        logger.info(
            "api_call_done",
            cache_key=key[:16],
            latency_sec=result["latency_sec"],
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
        )
        return result

    # ── internals ─────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(TransientAPIError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Call the API with tenacity retry on transient errors."""
        try:
            if self._enable_thinking:
                return self._call_thinking_stream(messages, max_tokens=max_tokens)
            return self._call_non_thinking(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except (TransientAPIError, PermanentAPIError):
            raise
        except Exception as exc:
            classified = _classify_api_error(exc)
            logger.warning(
                "api_error",
                error_type=type(classified).__name__,
                error=self._redact(str(exc)),
            )
            raise classified from exc

    def _call_non_thinking(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Non-thinking (synchronous, non-streaming) API call."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"enable_thinking": False},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        returned_model = getattr(response, "model", None) or self._model
        usage_obj = getattr(response, "usage", None)
        usage = (
            usage_obj.model_dump(exclude_none=True)
            if usage_obj is not None and hasattr(usage_obj, "model_dump")
            else {}
        )

        return {
            "content": content,
            "model": returned_model,
            "usage": _normalize_usage(usage),
            "has_reasoning_content": False,
        }

    def _call_thinking_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Thinking-mode (streaming) API call.

        WARNING: this path is not audit-safe (no temperature sent -> not
        deterministic; streaming returns no usage -> zero token counts in the
        cost ledger). It is gated by ``allow_thinking_mode`` in the model
        config; see the guard in ``__init__``.
        """
        if not self._allow_thinking_mode:
            # Defense in depth: __init__ already refuses to construct a
            # thinking-mode client without the explicit opt-in flag.
            raise NotImplementedError(
                "Thinking (streaming) path is disabled: non-deterministic "
                "(no temperature=0) and returns empty usage. Set "
                "'allow_thinking_mode: true' in the model config to override."
            )
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": True},
        )

        returned_model: str | None = None
        content_parts: list[str] = []
        has_reasoning_content = False

        for chunk in stream:
            returned_model = returned_model or getattr(chunk, "model", None)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # detect reasoning tokens
            if _chunk_has_reasoning(delta):
                has_reasoning_content = True

            delta_content = getattr(delta, "content", None)
            if delta_content:
                content_parts.append(delta_content)

        return {
            "content": "".join(content_parts),
            "model": returned_model or self._model,
            "usage": _normalize_usage({}),  # streaming doesn't return usage
            "has_reasoning_content": has_reasoning_content,
        }

    def _redact(self, text: str) -> str:
        """Remove API key from error messages."""
        if self._api_key:
            return text.replace(self._api_key, "[REDACTED]")
        return text


# ── Helpers ───────────────────────────────────────────────────────────────


def _chunk_has_reasoning(delta: Any) -> bool:
    """Check whether a streaming delta chunk contains reasoning tokens."""
    if getattr(delta, "reasoning_content", None):
        return True
    try:
        d = delta.model_dump(exclude_none=True)
    except AttributeError:
        return False
    return any(d.get(k) for k in ("reasoning_content", "reasoning", "thinking"))


def _normalize_usage(raw: dict[str, Any]) -> dict[str, int]:
    """Normalize DashScope / OpenAI usage dicts to a common shape."""
    return {
        "prompt_tokens": raw.get("prompt_tokens") or raw.get("input_tokens") or 0,
        "completion_tokens": (
            raw.get("completion_tokens") or raw.get("output_tokens") or 0
        ),
        "total_tokens": raw.get("total_tokens") or 0,
    }
