"""Anthropic's native Messages wire: the listing's thinking shapes and the levels they take."""

import pytest

from otaku.providers import ProviderConfig, ProviderError
from otaku.providers.clients.anthropic import (
    AnthropicClient,
    Thinking,
    _knobs,
    _stated_thinking,
    capabilities,
    spec_of,
)

_ADAPTIVE_AND_OFF = {
    "thinking": {
        "supported": True,
        "types": {
            "adaptive": {"supported": True},
            "enabled": {"supported": False},
            "disabled": {"supported": True},
        },
    },
    "effort": {
        "supported": True,
        "low": {"supported": True},
        "medium": {"supported": True},
        "high": {"supported": True},
        "max": {"supported": True},
        "xhigh": None,
    },
}
_MANUAL_ONLY = {
    "thinking": {
        "supported": True,
        "types": {
            "adaptive": {"supported": False},
            "enabled": {"supported": True},
            "disabled": {"supported": True},
        },
    },
    "effort": {"supported": False},
}


class TestStatedThinking:
    def test_adaptive_rungs_are_the_supported_efforts_and_off_is_a_rung(self) -> None:
        thinking = _stated_thinking(_ADAPTIVE_AND_OFF)
        assert thinking == Thinking(
            efforts=frozenset({"low", "medium", "high", "max"}), disabled=True
        )
        caps = capabilities(thinking)
        assert caps.reasoning_efforts == {"none", "low", "medium", "high", "max"}
        assert caps.reasoning_switch is False
        assert caps.reasoning_budget is False

    def test_a_budget_model_is_a_switch_with_a_budget(self) -> None:
        caps = capabilities(_stated_thinking(_MANUAL_ONLY))
        assert not caps.reasoning_efforts
        assert caps.reasoning_switch is True
        assert caps.reasoning_budget is True

    def test_a_model_without_thinking_takes_no_level(self) -> None:
        thinking = _stated_thinking({"thinking": {"supported": False}})
        with pytest.raises(ProviderError):
            _knobs("m", "high", thinking, 4096)


class TestKnobs:
    def test_a_rung_is_adaptive_thinking_at_that_effort(self) -> None:
        assert _knobs("m", "high", _stated_thinking(_ADAPTIVE_AND_OFF), 4096) == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }

    @pytest.mark.parametrize("level", ["none", "off", "0"])
    def test_off_disables_thinking_where_the_model_allows_it(self, level: str) -> None:
        assert _knobs("m", level, _stated_thinking(_ADAPTIVE_AND_OFF), 4096) == {
            "thinking": {"type": "disabled"}
        }

    def test_a_budget_is_manual_thinking_bounded_by_the_reply_cap(self) -> None:
        thinking = _stated_thinking(_MANUAL_ONLY)
        assert _knobs("m", "2048", thinking, 4096) == {
            "thinking": {"type": "enabled", "budget_tokens": 2048}
        }
        assert _knobs("m", "on", thinking, 4096) == {
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        }
        for level, cap in (("512", 4096), ("4096", 4096)):
            with pytest.raises(ProviderError):
                _knobs("m", level, thinking, cap)

    def test_a_rung_is_refused_on_a_budget_model(self) -> None:
        with pytest.raises(ProviderError):
            _knobs("m", "high", _stated_thinking(_MANUAL_ONLY), 4096)


class TestKnownModels:
    def test_a_dated_snapshot_reads_as_its_alias(self) -> None:
        assert spec_of("claude-haiku-4-5-20251001") == spec_of("claude-haiku-4-5")
        assert spec_of("claude-haiku-4-5") is not None
        assert spec_of("claude-unknown-1") is None


class TestClient:
    def test_the_section_founded_names_anthropics_endpoint(self) -> None:
        config = AnthropicClient.autoconfigure()
        client = AnthropicClient(_with_key(config))
        assert client.completion._chat_url() == "https://api.anthropic.com/v1/messages"
        assert client.auth.headers == {"x-api-key": "k", "anthropic-version": "2023-06-01"}


def _with_key(config: ProviderConfig) -> ProviderConfig:
    return ProviderConfig(name=config.name, url=config.url, api_key="k")
