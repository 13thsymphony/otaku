"""Azure-hosted Claude via Foundry's native Anthropic Messages API.

Deployments are configured explicitly: v2 exposes Messages and token
counting, not a models catalog. A count checks each deployment without
generating a reply. Native blocks and SSE use the shared completion
lifecycle, so cancellation, pacing and request logging remain identical.
Sampling controls are omitted; modern Claude models reject them.
"""

import base64
import json
from collections.abc import Generator, Iterator, Sequence
from dataclasses import replace
from typing import Any

from otaku.providers.errors import DeclinedError, ProviderError, UnauthorizedError, UnreachableError
from otaku.providers.http import ASK_TIMEOUT, ErrorSink, Http
from otaku.providers.openai import frames
from otaku.providers.openai.auth import OpenAIAuth
from otaku.providers.openai.client import OpenAIClient
from otaku.providers.openai.completion import OpenAICompletion, RequestSink
from otaku.providers.openai.models import Locality, ModelCapabilities, ModelInfo, OpenAIModels
from otaku.providers.openai.requests import WireMessage
from otaku.settings.providers import ProviderConfig

_PARAMS = frozenset({"max_tokens", "stop"})
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
# Azure-hosted adaptive models: context, output cap and supported efforts.
_MODELS = {
    name: (1_000_000, 128_000, _EFFORTS)
    for name in (
        "claude-opus-5-5",
        "claude-opus-5-5-fast",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
    )
}
_MODELS["claude-haiku-4-5"] = (200_000, 64_000, frozenset())


class FoundryAuth(OpenAIAuth):
    @property
    def headers(self) -> dict[str, str]:
        return {
            **({"x-api-key": self.api_key} if self.api_key else {}),
            "anthropic-version": "2023-06-01",
        }


class FoundryModels(OpenAIModels):
    def _list(self, http: Http) -> list[ModelInfo]:
        if not self._auth.api_key:
            raise UnauthorizedError(f"No api key for {self._config.name}.")
        if not self._config.deployments:
            raise ProviderError("Configure Foundry deployment names in providers.toml.")
        models = []
        for deployment, model_id in self._config.deployments.items():
            count = http.post(
                f"{self._config.url}/messages/count_tokens",
                {"model": deployment, "messages": [{"role": "user", "content": "Hi"}]},
            )
            if not isinstance(count, dict) or _count(count.get("input_tokens")) is None:
                raise ProviderError(f"Foundry returned no token count for {deployment}.")
            spec = _MODELS.get(model_id)
            models.append(
                ModelInfo(
                    deployment,
                    context_length=spec[0] if spec else None,
                    max_output_tokens=spec[1] if spec else None,
                    capabilities=ModelCapabilities(
                        vision=True,
                        audio=False,
                        text_completion=False,
                        structured_output=True,
                        supported_params=_PARAMS,
                        reasoning_efforts=spec[2] if spec else None,
                        reasoning_switch=model_id == "claude-haiku-4-5",
                        reasoning_budget=model_id == "claude-haiku-4-5",
                    ),
                )
            )
        return models

    def _get(self, name: str, http: Http) -> ModelInfo | None:
        return None


class FoundryCompletion(OpenAICompletion):
    supported_params = _PARAMS
    can_mark_cache = True
    can_count_tokens = True
    chat_reasoning_knobs = frozenset()

    def _chat_url(self) -> str:
        return f"{_base_url(self._config.url)}/messages"

    def _chat_request(
        self,
        model: str,
        messages: Sequence[WireMessage],
        params: dict[str, object],
        *,
        level: str | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if model not in self._config.deployments:
            raise ProviderError(f"No Foundry deployment is configured as {model!r}.")
        wire_params = self._wire_params(params, model)
        if "stop" in wire_params:
            wire_params["stop_sequences"] = wire_params.pop("stop")
        wire_params.setdefault("max_tokens", 4096)
        system, wire = _blocks(messages, self._cache_ttl())
        body: dict[str, object] = {
            "model": model,
            "messages": wire,
            "stream": True,
            **wire_params,
        }
        if system:
            body["system"] = system
        knobs: dict[str, object] = {}
        if level is not None:
            model_id = self._config.deployments[model]
            spec = _MODELS.get(model_id)
            if model_id == "claude-haiku-4-5":
                knobs = _legacy_thinking(level, wire_params["max_tokens"])
            elif spec is not None and level in spec[2]:
                knobs = {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
            else:
                raise ProviderError(
                    f"Foundry deployment {model!r} does not support effort {level!r}."
                )
        return body, knobs

    def count_chat_tokens(
        self,
        model: str,
        messages: Sequence[WireMessage],
        *,
        level: str | None = None,
        timeout: float = ASK_TIMEOUT,
    ) -> int | None:
        body, knobs = self._chat_request(model, messages, {}, level=level)
        # Effort and max_tokens do not affect input token counting.
        body.pop("stream")
        body.pop("max_tokens")
        if "thinking" in knobs:
            body["thinking"] = knobs["thinking"]
        data = self._http.post(
            f"{self._chat_url()}/count_tokens",
            body,
            purpose="count",
            timeout=timeout,
        )
        count = _count(data.get("input_tokens")) if isinstance(data, dict) else None
        if count is None:
            raise ProviderError("Foundry returned no input token count.")
        return count

    @staticmethod
    def _chat_delta(event: dict[str, Any]) -> tuple[str, str]:
        if event.get("type") == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict):
                if block.get("type") == "text":
                    return "", str(block.get("text") or "")
                if block.get("type") == "thinking":
                    return str(block.get("thinking") or ""), ""
        if event.get("type") != "content_block_delta":
            return "", ""
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return "", ""
        if delta.get("type") == "thinking_delta":
            return str(delta.get("thinking") or ""), ""
        if delta.get("type") == "text_delta":
            return "", str(delta.get("text") or "")
        return "", ""

    @staticmethod
    def _finish_reason(event: dict[str, Any]) -> str | None:
        delta = event.get("delta")
        reason = delta.get("stop_reason") if isinstance(delta, dict) else None
        if reason == "refusal":
            raise DeclinedError("The model declined the request.")
        if reason == "max_tokens":
            return "length"
        if reason in ("end_turn", "stop_sequence"):
            return "stop"
        return reason if isinstance(reason, str) else None

    def _read_usage(self, event: dict[str, Any]) -> frames.Usage | None:
        message = event.get("message")
        usage = message.get("usage") if isinstance(message, dict) else event.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = _count(usage.get("input_tokens"))
        cached = _count(usage.get("cache_read_input_tokens"))
        created = _count(usage.get("cache_creation_input_tokens"))
        if prompt is not None:
            prompt += (cached or 0) + (created or 0)
        return prompt, _count(usage.get("output_tokens")), cached

    @staticmethod
    def _events(lines: Iterator[str], name: str) -> Generator[dict[str, Any], None, None]:
        for line in lines:
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:") :].strip())
            except json.JSONDecodeError as e:
                raise ProviderError("Foundry returned an invalid streaming event.") from e
            if not isinstance(event, dict):
                raise ProviderError("Foundry returned a streaming event that is not an object.")
            if event.get("type") == "message_stop":
                return
            yield event
        raise UnreachableError(f"Lost the connection to {name}.")


class FoundryClient(OpenAIClient):
    id = "foundry"
    label = "Azure Foundry Claude"
    locality = Locality.REMOTE
    env_key = "ANTHROPIC_FOUNDRY_API_KEY"
    auth_class = FoundryAuth
    models_class = FoundryModels
    completion_class = FoundryCompletion

    def __init__(
        self,
        config: ProviderConfig,
        *,
        request_sink: RequestSink | None = None,
        error_sink: ErrorSink | None = None,
        smooth: bool = False,
    ) -> None:
        super().__init__(
            replace(config, url=_base_url(config.url)),
            request_sink=request_sink,
            error_sink=error_sink,
            smooth=smooth,
        )


def _base_url(url: str) -> str:
    root = url.rstrip("/").removesuffix("/messages").removesuffix("/v1")
    return f"{root}/v1"


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _legacy_thinking(level: str, max_tokens: object) -> dict[str, object]:
    if level in ("none", "off", "0"):
        return {"thinking": {"type": "disabled"}}
    budget = 1024 if level == "on" else int(level) if level.isdecimal() else None
    cap = _count(max_tokens)
    if budget is None or budget < 1024 or cap is None or budget >= cap:
        raise ProviderError("Haiku 4.5 needs a thinking budget of at least 1024, below max_tokens.")
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _blocks(
    messages: Sequence[WireMessage],
    cache_ttl: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    system: list[dict[str, object]] = []
    wire: list[dict[str, object]] = []
    stable: list[dict[str, object]] = []
    is_stable = True
    for message in messages:
        is_stable = is_stable and not message.volatile
        parts: list[dict[str, object]] = []
        if message.body:
            parts.append({"type": "text", "text": message.body})
        for image in message.images:
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    },
                }
            )
        if message.role == "system":
            if wire:
                raise ProviderError("Foundry requires system instructions before the conversation.")
            if message.images:
                raise ProviderError("Foundry does not accept images in system instructions.")
            system.extend(parts)
        else:
            role = "user" if message.role == "tool" else message.role
            if role not in ("user", "assistant"):
                raise ProviderError(f"Foundry does not support message role {message.role!r}.")
            if parts:
                # Anthropic merges consecutive turns of the same role.
                wire.append({"role": role, "content": parts})
        if parts and is_stable:
            stable.append(parts[-1])
    if cache_ttl:
        marker: dict[str, object] = {"type": "ephemeral"}
        if cache_ttl == "1h":
            marker["ttl"] = "1h"
        if system and any(part is system[-1] for part in stable):
            system[-1]["cache_control"] = dict(marker)
        if stable:
            stable[-1]["cache_control"] = dict(marker)
    # New Claude models reject assistant prefills. An empty submission
    # continues the conversation through an explicit user request.
    if not wire or wire[-1]["role"] == "assistant":
        wire.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    return system, wire
