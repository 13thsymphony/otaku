"""Foundry's native Messages wire: deployment routing, blocks and SSE."""

import json
from dataclasses import dataclass

import pytest

from otaku.providers import Image, ProviderConfig, ProviderError, UnreachableError
from otaku.providers.clients.foundry import FoundryClient, FoundryCompletion
from otaku.providers.http import Http


class TestRequest:
    def test_deployment_system_images_and_effort_use_the_anthropic_wire(self) -> None:
        completion = _completion()
        body, knobs = completion._chat_request(
            "my-opus",
            [_Turn("system", "Premise"), _Turn("user", "Look", (Image(b"png", "image/png"),))],
            {"max_tokens": 4096, "temperature": 0.7, "seed": 42},
            level="high",
        )
        assert body["model"] == "my-opus"
        assert body["system"] == [
            {"type": "text", "text": "Premise", "cache_control": {"type": "ephemeral"}}
        ]
        assert body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "cG5n",
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
        assert body["max_tokens"] == 4096
        assert body["stream"] is True
        assert not {"temperature", "seed", "stream_options"} & body.keys()
        assert knobs == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}

    def test_volatile_rows_and_everything_after_them_are_not_cache_marked(self) -> None:
        body, _ = _completion()._chat_request(
            "my-opus",
            [_Turn("system", "s"), _Turn("user", "changing", volatile=True), _Turn("user", "u")],
            {},
            level=None,
        )
        assert "cache_control" not in json.dumps(body["messages"])

    def test_stop_is_respelled_and_the_required_output_cap_has_a_default(self) -> None:
        body, _ = _completion()._chat_request(
            "my-opus", [_Turn("user", "u")], {"stop": ["END"]}, level=None
        )
        assert body["stop_sequences"] == ["END"]
        assert isinstance(body["max_tokens"], int) and body["max_tokens"] > 0

    @pytest.mark.parametrize("level", ["none", "off", "0", "1024", "minimal", "typo"])
    def test_opus_55_never_sends_disabled_thinking_or_a_manual_budget(self, level: str) -> None:
        with pytest.raises(ProviderError):
            _completion()._chat_request("my-opus", [_Turn("user", "u")], {}, level=level)

    @pytest.mark.parametrize("suffix", ["", "/v1", "/v1/messages", "/v1/messages/"])
    def test_target_uris_and_base_urls_reach_the_same_messages_endpoint(self, suffix: str) -> None:
        client = FoundryClient(_config("https://resource.services.ai.azure.com/anthropic" + suffix))
        assert client.completion._chat_url() == (
            "https://resource.services.ai.azure.com/anthropic/v1/messages"
        )
        assert client.auth.headers == {
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
        }

    def test_a_continuation_does_not_send_an_assistant_prefill(self) -> None:
        body, _ = _completion()._chat_request(
            "my-opus", [_Turn("user", "u"), _Turn("assistant", "a")], {}, level=None
        )
        assert body["messages"][-1] == {
            "role": "user",
            "content": [{"type": "text", "text": "Continue."}],
        }

    def test_haiku_45_uses_a_bounded_manual_budget_instead_of_adaptive_thinking(self) -> None:
        config = ProviderConfig(
            name="foundry",
            url="https://example/anthropic",
            deployments={"haiku": "claude-haiku-4-5"},
        )
        completion = FoundryCompletion(config, Http("foundry", {}), request_sink=None, smooth=False)
        _, knobs = completion._chat_request("haiku", [_Turn("user", "u")], {}, level="1024")
        assert knobs == {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        _, knobs = completion._chat_request("haiku", [_Turn("user", "u")], {}, level="off")
        assert knobs == {"thinking": {"type": "disabled"}}
        with pytest.raises(ProviderError):
            completion._chat_request(
                "haiku", [_Turn("user", "u")], {"max_tokens": 1024}, level="1024"
            )


class TestEvents:
    def test_message_stop_is_the_clean_end_not_openais_done_marker(self) -> None:
        event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}
        lines = iter(
            [
                "event: content_block_delta",
                "data: " + json.dumps(event),
                'data: {"type":"message_stop"}',
            ]
        )
        assert list(FoundryCompletion._events(lines, "foundry")) == [event]
        assert FoundryCompletion._chat_delta(event) == ("", "Hi")

    def test_thinking_is_not_text_and_signature_deltas_are_not_displayed(self) -> None:
        event = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Think"},
        }
        assert FoundryCompletion._chat_delta(event) == ("Think", "")
        assert FoundryCompletion._chat_delta(
            {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "s"}}
        ) == ("", "")
        assert FoundryCompletion._chat_delta(
            {"type": "content_block_start", "content_block": {"type": "text", "text": "First"}}
        ) == ("", "First")

    def test_usage_includes_cache_reads_and_writes_and_keeps_zero_counts(self) -> None:
        completion = _completion()
        assert completion._read_usage(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 30,
                    }
                },
            }
        ) == (60, 0, 20)
        assert completion._read_usage({"type": "message_delta", "usage": {"output_tokens": 7}}) == (
            None,
            7,
            None,
        )
        assert (
            completion._finish_reason(
                {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}
            )
            == "length"
        )

    def test_a_truncated_stream_is_not_a_success(self) -> None:
        with pytest.raises(UnreachableError):
            list(FoundryCompletion._events(iter(['data: {"type":"ping"}']), "foundry"))


@dataclass(frozen=True)
class _Turn:
    role: str
    body: str
    images: tuple[Image, ...] = ()
    volatile: bool = False


def _config(url: str = "https://resource.services.ai.azure.com/anthropic") -> ProviderConfig:
    return ProviderConfig(
        name="foundry",
        url=url,
        api_key="test-key",
        deployments={"my-opus": "claude-opus-5-5"},
    )


def _completion() -> FoundryCompletion:
    return FoundryCompletion(_config(), Http("foundry", {}), request_sink=None, smooth=False)
