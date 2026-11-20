"""Claude through Anthropic's native Messages API, at api.anthropic.com.

The Models API lists each model with its context, output cap, image
input and thinking shapes — adaptive with effort rungs, a manual budget,
or off — so a listing carries the facts a request is shaped by; a table
of known models stands in where an entry states none. A level goes out
as the model takes it: a rung as adaptive thinking with that effort, a
budget as manual thinking, off as disabled thinking. Native blocks and
SSE use the shared completion lifecycle, so cancellation, pacing and
request logging stay identical; sampling controls are omitted, as modern
Claude models reject them. Foundry's Claude (`clients.foundry`) speaks
this same wire behind deployment names.
"""

import base64
import json
import re
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import quote

from otaku.providers.errors import DeclinedError, ProviderError, UnauthorizedError, UnreachableError
from otaku.providers.http import ASK_TIMEOUT, ErrorSink, Http, positive_int
from otaku.providers.openai import frames
from otaku.providers.openai.auth import OpenAIAuth
from otaku.providers.openai.client import OpenAIClient
from otaku.providers.openai.completion import OpenAICompletion, RequestSink
from otaku.providers.openai.models import Locality, ModelCapabilities, ModelInfo, OpenAIModels
from otaku.providers.openai.requests import WireMessage
from otaku.settings.providers import ProviderConfig

_URL = "https://api.anthropic.com/v1"
_PARAMS = frozenset({"max_tokens", "stop"})
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_OFF = frozenset({"none", "off", "0"})
_DATED = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class Thinking:
    """How a model thinks: the effort rungs adaptive thinking grades,
    whether it may be switched off, whether a manual budget holds."""

    efforts: frozenset[str] = frozenset()
    can_disable: bool = False
    can_set_budget: bool = False


@dataclass(frozen=True)
class Spec:
    """A known model's facts, for an entry or a deployment that states none."""

    context: int
    output: int
    thinking: Thinking


_ADAPTIVE = Thinking(efforts=frozenset(_EFFORTS))
_ADAPTIVE_OFF = Thinking(efforts=frozenset(_EFFORTS), can_disable=True)
_MANUAL = Thinking(can_disable=True, can_set_budget=True)
_MODELS = {
    **{
        name: Spec(1_000_000, 128_000, _ADAPTIVE)
        for name in (
            "claude-opus-5-5",
            "claude-opus-5-5-fast",
            "claude-sonnet-5-5",
        )
    },
    **{
        name: Spec(1_000_000, 128_000, _ADAPTIVE_OFF)
        for name in (
            "claude-opus-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-5-5",
        )
    },
    "claude-haiku-4-5": Spec(200_000, 64_000, _MANUAL),
}


def spec_of(model_id: str) -> Spec | None:
    """The known facts of `model_id`, a dated snapshot read as its alias."""
    return _MODELS.get(model_id) or _MODELS.get(_DATED.sub("", model_id))


def model_info(name: str, spec: Spec | None) -> ModelInfo:
    """A listing row for a model that thinks as `spec` says."""
    thinking = spec.thinking if spec else None
    known = True if spec else None
    return ModelInfo(
        name,
        context_length=spec.context if spec else None,
        max_output_tokens=spec.output if spec else None,
        capabilities=capabilities(thinking, vision=known, structured=known),
    )


def capabilities(
    thinking: Thinking | None, *, vision: bool | None = None, structured: bool | None = None
) -> ModelCapabilities:
    efforts = thinking.efforts if thinking else None
    return ModelCapabilities(
        vision=vision,
        audio=False,
        text_completion=False,
        structured_output=structured,
        supported_params=_PARAMS,
        reasoning_efforts=(
            efforts | {"none"} if efforts and thinking and thinking.can_disable else efforts
        ),
        reasoning_switch=(
            thinking.can_disable and not thinking.efforts if thinking is not None else None
        ),
        reasoning_budget=thinking.can_set_budget if thinking is not None else None,
    )


class AnthropicAuth(OpenAIAuth):
    @property
    def headers(self) -> dict[str, str]:
        return {
            **({"x-api-key": self.api_key} if self.api_key else {}),
            "anthropic-version": "2023-06-01",
        }


class AnthropicModels(OpenAIModels):
    def __init__(self, config: ProviderConfig, auth: OpenAIAuth, http: Http) -> None:
        super().__init__(config, auth, http)
        self.thinking: dict[str, Thinking] = {}

    def _list(self, http: Http) -> list[ModelInfo]:
        if not self._auth.api_key:
            raise UnauthorizedError(f"No api key for {self._config.name}.")
        models: list[ModelInfo] = []
        after = ""
        cursors: set[str] = set()
        while True:
            data = http.get(f"{self._config.url}/models?limit=1000{after}")
            entries = data.get("data") if isinstance(data, dict) else None
            has_more = data.get("has_more") if isinstance(data, dict) else None
            if not isinstance(entries, list) or not isinstance(has_more, bool):
                raise ProviderError(f"{self._config.name} returned an invalid models page.")
            if has_more:
                last = data.get("last_id")
                if not isinstance(last, str) or not last or last in cursors:
                    raise ProviderError(f"{self._config.name} returned an invalid models cursor.")
                cursors.add(last)
                after = f"&after_id={quote(last, safe='')}"
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("id"), str)
                    or not entry["id"]
                ):
                    raise ProviderError(f"{self._config.name} returned an invalid model.")
                models.append(self._entry(entry))
            if not has_more:
                break
        return sorted(models, key=lambda model: model.name)

    def _entry(self, entry: dict[str, Any]) -> ModelInfo:
        name = str(entry["id"])
        spec = spec_of(name)
        stated = entry.get("capabilities")
        if not isinstance(stated, dict):
            model = model_info(name, spec)
            if spec:
                self.thinking[name] = spec.thinking
        else:
            thinking = _stated_thinking(stated)
            self.thinking[name] = thinking
            model = ModelInfo(
                name,
                capabilities=capabilities(
                    thinking,
                    vision=_supported(stated.get("image_input")),
                    structured=_supported(stated.get("structured_outputs")),
                ),
            )
        return replace(
            model,
            context_length=positive_int(entry.get("max_input_tokens")) or model.context_length,
            max_output_tokens=positive_int(entry.get("max_tokens")) or model.max_output_tokens,
        )


class AnthropicCompletion(OpenAICompletion):
    supported_params = _PARAMS
    can_mark_cache = True
    can_count_tokens = True
    chat_reasoning_knobs = frozenset()

    def _chat_url(self) -> str:
        return f"{self._config.url}/messages"

    @staticmethod
    def _refusal_words(knobs: dict[str, object]) -> set[str]:
        # "type" and "display" occur in unrelated request fields, so
        # neither alone proves that thinking was refused.
        return OpenAICompletion._refusal_words(knobs) - {"type", "display"}

    def _thinking_of(self, model: str) -> Thinking | None:
        """How `model` thinks: as listed, else as a known model."""
        if isinstance(self._models, AnthropicModels) and model in self._models.thinking:
            return self._models.thinking[model]
        spec = spec_of(model)
        return spec.thinking if spec else None

    def _validate_model(self, model: str) -> None:
        """Refuse a model this provider cannot route; any name is sent here."""

    def _chat_request(
        self,
        model: str,
        messages: Sequence[WireMessage],
        params: dict[str, object],
        *,
        level: str | None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self._validate_model(model)
        wire_params = self._wire_params(params, model)
        if "stop" in wire_params:
            wire_params["stop_sequences"] = wire_params.pop("stop")
        wire_params.setdefault("max_tokens", 4096)
        max_tokens = _count(wire_params["max_tokens"])
        if max_tokens is None:
            raise ProviderError("Claude max_tokens must be a nonnegative integer.")
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
            knobs = _knobs(model, level, self._thinking_of(model), max_tokens)
        return body, knobs

    def count_chat_tokens(
        self,
        model: str,
        messages: Sequence[WireMessage],
        *,
        level: str | None = None,
        timeout: float = ASK_TIMEOUT,
    ) -> int | None:
        body, _ = self._chat_request(model, messages, {}, level=None)
        # Effort and max_tokens do not affect input token counting.
        body.pop("stream")
        body.pop("max_tokens")
        if level is not None:
            knobs = _knobs(model, level, self._thinking_of(model), None)
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
            raise ProviderError(f"{self._config.name} returned no input token count.")
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
        if reason in ("max_tokens", "model_context_window_exceeded"):
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
                raise ProviderError(f"{name} returned an invalid streaming event.") from e
            if not isinstance(event, dict):
                raise ProviderError(f"{name} returned a streaming event that is not an object.")
            if event.get("type") == "message_stop":
                return
            yield event
        raise UnreachableError(f"Lost the connection to {name}.")


class AnthropicClient(OpenAIClient):
    id = "anthropic"
    label = "Anthropic"
    locality = Locality.REMOTE
    env_key = "ANTHROPIC_API_KEY"
    auth_class = AnthropicAuth
    models_class = AnthropicModels
    completion_class = AnthropicCompletion

    def __init__(
        self,
        config: ProviderConfig,
        *,
        request_sink: RequestSink | None = None,
        error_sink: ErrorSink | None = None,
        smooth: bool = False,
    ) -> None:
        super().__init__(
            replace(config, url=_base_url(config.url)) if config.url else config,
            request_sink=request_sink,
            error_sink=error_sink,
            smooth=smooth,
        )

    @classmethod
    def autoconfigure(cls) -> ProviderConfig:
        # The deliberate-add default: Anthropic's one endpoint; the api
        # key is the user's to provide.
        return ProviderConfig(name=cls.id, url=_URL)


def _base_url(url: str) -> str:
    """The /v1 root, whether the url names it, the host or the Messages endpoint."""
    root = url.rstrip("/").removesuffix("/messages").removesuffix("/v1")
    return f"{root}/v1"


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _supported(value: object) -> bool | None:
    return value.get("supported") is True if isinstance(value, dict) else None


def _stated_thinking(stated: dict[str, Any]) -> Thinking:
    """A Models API entry's thinking shapes, as a request is built from them."""
    thinking = stated.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("supported") is not True:
        return Thinking()
    types = thinking.get("types")
    types = types if isinstance(types, dict) else {}
    effort = stated.get("effort")
    efforts: frozenset[str] = frozenset()
    if _supported(types.get("adaptive")) and isinstance(effort, dict) and _supported(effort):
        efforts = frozenset(level for level in _EFFORTS if _supported(effort.get(level)))
    return Thinking(
        efforts=efforts,
        can_disable=_supported(types.get("disabled")) is True,
        can_set_budget=_supported(types.get("enabled")) is True,
    )


def _knobs(
    model: str, level: str, thinking: Thinking | None, max_tokens: int | None
) -> dict[str, object]:
    if thinking is not None:
        if level in thinking.efforts:
            return {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": level},
            }
        if level in _OFF and thinking.can_disable:
            return {"thinking": {"type": "disabled"}}
        if thinking.can_set_budget and (level == "on" or level.isdecimal()):
            budget = 1024 if level == "on" else int(level)
            if budget < 1024 or (max_tokens is not None and budget >= max_tokens):
                raise ProviderError(
                    f"{model} needs a thinking budget of at least 1024, below max_tokens."
                )
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    raise ProviderError(f"{model} does not support thinking level {level!r}.")


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
                raise ProviderError("Claude requires system instructions before the conversation.")
            if message.images:
                raise ProviderError("Claude does not accept images in system instructions.")
            system.extend(parts)
        else:
            role = "user" if message.role == "tool" else message.role
            if role not in ("user", "assistant"):
                raise ProviderError(f"Claude does not support message role {message.role!r}.")
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
