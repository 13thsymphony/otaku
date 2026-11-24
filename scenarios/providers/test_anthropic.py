"""Anthropic's own API through the real provider, app and native HTTP wire."""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from otaku.backend import launch as backend_launch
from otaku.backend.api import Api, ModelRef, ModelRole
from otaku.backend.app.paths import Paths
from otaku.providers import ProviderConfig, ProviderError, Reasoning, Text, autoconfigure
from otaku.providers.clients.anthropic import AnthropicClient
from otaku.settings import providers as providers_file
from otaku.settings import write_atomic
from scenarios.support.harness import set_config_provider
from scenarios.support.server import ModelServer

_OPUS = {
    "id": "claude-opus-5-5",
    "type": "model",
    "max_input_tokens": 900_000,
    "max_tokens": 64_000,
    "capabilities": {
        "image_input": {"supported": True},
        "structured_outputs": {"supported": True},
        "thinking": {
            "supported": True,
            "types": {
                "adaptive": {"supported": True},
                "enabled": {"supported": False},
                "disabled": {"supported": False},
            },
        },
        "effort": {
            "supported": True,
            "low": {"supported": True},
            "medium": {"supported": True},
            "high": {"supported": True},
            "max": {"supported": True},
            "xhigh": {"supported": True},
        },
    },
}
# An entry from an API version that states no capabilities: the known
# model's facts stand in.
_HAIKU = {"id": "claude-haiku-4-5-20251001", "type": "model", "capabilities": None}


class TestListing:
    def test_gateway_model_lists_use_their_stated_context_without_pagination(
        self, server: ModelServer
    ) -> None:
        client = _client(server)
        server.anthropic_model_list = True
        server.anthropic_models = [
            {"id": "claude-fable-5-2", "context_length": 1_000_000},
            {"id": "claude-haiku-4-5-20251001", "context_length": 100_000},
        ]
        fable, haiku = client.models.list()
        assert fable.context_length == fable.context_window == 1_000_000
        assert fable.capabilities.vision is None
        assert fable.capabilities.reasoning_efforts is None
        assert haiku.context_length == haiku.context_window == 100_000
        assert len(server.gets) == 1

    def test_every_page_is_listed_with_the_facts_each_entry_states(
        self, server: ModelServer
    ) -> None:
        client = _client(server)
        haiku, opus = client.models.list()
        assert server.gets == ["/anthropic/v1/models", "/anthropic/v1/models"]
        assert (opus.name, opus.context_length, opus.max_output_tokens) == (
            "claude-opus-5-5",
            900_000,
            64_000,
        )
        assert opus.capabilities.reasoning_efforts == {"low", "medium", "high", "xhigh", "max"}
        assert opus.capabilities.vision is True
        assert haiku.name == "claude-haiku-4-5-20251001"
        assert haiku.context_length == 200_000
        assert haiku.capabilities.reasoning_budget is True

    @pytest.mark.parametrize("missing", [True, False])
    def test_a_missing_or_repeated_cursor_fails_instead_of_truncating_the_listing(
        self, server: ModelServer, missing: bool
    ) -> None:
        client = _client(server)
        server.anthropic_cursor_missing = missing
        server.anthropic_models = [{"id": name} for name in ("same", "same", "third")]
        with pytest.raises(ProviderError):
            client.models.list(timeout=1.0)
        assert len(server.gets) <= 2

    def test_a_cursor_with_reserved_characters_is_sent_as_one_query_value(
        self, server: ModelServer
    ) -> None:
        client = _client(server)
        server.anthropic_models = [{"id": "claude+legacy"}, {"id": "claude-new"}]
        assert [model.name for model in client.models.list(timeout=1.0)] == [
            "claude+legacy",
            "claude-new",
        ]
        assert len(server.gets) == 2

    def test_a_listed_model_thinks_as_listed(self, server: ModelServer) -> None:
        client = _client(server)
        client.models.list()
        server.script = lambda _: ("Thought", "Answer")
        chunks = list(
            client.completion.chat("claude-opus-5-5", [_Turn("user", "Hi")], {}, level="xhigh")
        )
        assert "".join(c.text for c in chunks if isinstance(c, Reasoning)) == "Thought"
        assert "".join(c.text for c in chunks if isinstance(c, Text)) == "Answer"
        sent = server.requests[-1]
        assert sent["model"] == "claude-opus-5-5"
        assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert sent["output_config"] == {"effort": "xhigh"}
        list(
            client.completion.chat(
                "claude-haiku-4-5-20251001",
                [_Turn("user", "Hi")],
                {"max_tokens": 4096},
                level="2048",
            )
        )
        assert server.requests[-1]["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_counting_uses_the_messages_count_endpoint(self, server: ModelServer) -> None:
        client = _client(server)
        assert client.completion.count_chat_tokens("claude-opus-5-5", [_Turn("user", "Hi")]) == 17
        assert server.posts[-1] == "/anthropic/v1/messages/count_tokens"
        assert server.request_headers[-1]["x-api-key"] == "fake-key"
        assert server.request_headers[-1]["anthropic-version"] == "2023-06-01"

    def test_counting_accepts_a_budget_larger_than_the_default_reply_limit(
        self, server: ModelServer
    ) -> None:
        client = _client(server)
        assert (
            client.completion.count_chat_tokens(
                "claude-haiku-4-5", [_Turn("user", "Hi")], level="8192"
            )
            == 17
        )
        sent = server.requests[-1]
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        assert "max_tokens" not in sent


class TestSetup:
    def test_the_api_key_variable_founds_the_section(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        founded = autoconfigure()
        assert founded["anthropic"].url == "https://api.anthropic.com/v1"
        assert "foundry" not in founded

    @pytest.mark.parametrize("provider", ["anthropic", "generic_anthropic"])
    def test_the_real_app_plays_and_persists_an_anthropic_reply(
        self, tmp_path: Path, server: ModelServer, provider: str
    ) -> None:
        root = tmp_path / "state"
        set_config_provider(root, server)
        backend_launch.open_app(root).close()
        paths = Paths.resolve(root)
        written = providers_file.load(paths.providers_file)
        written[provider] = replace(_config(server), name=provider)
        write_atomic(paths.providers_file, providers_file.render(written))
        backend = backend_launch.open_app(root)
        try:
            api = Api(backend)
            api.providers.list(scope=provider)
            api.app.update(models={ModelRole.WRITING: ModelRef(provider, "claude-opus-5-5")})
            settings = api.providers.models.settings.get(provider, "claude-opus-5-5")
            assert set(settings.choices.levels) == {
                "unset",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }
            assert not settings.choices.budget
            api.providers.models.settings.update(provider, "claude-opus-5-5", think="high")
            story = api.stories.create().value
            assert list(
                api.stories.messages.create(
                    story.id, line="Open the gate.", expected_head=story.head
                )
            )
            messages = api.stories.messages.list(story.id)
            assert messages[-1].body == "The light went out, and something stirred in the dark."
            assert "/anthropic/v1/messages" in server.posts
            sent = next(body for body in server.requests if body.get("stream") is True)
            assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert sent["output_config"] == {"effort": "high"}
        finally:
            backend.close()

    def test_custom_endpoint_edits_leave_the_other_providers_unchanged(
        self, tmp_path: Path, server: ModelServer
    ) -> None:
        root = tmp_path / "state"
        set_config_provider(root, server)
        backend_launch.open_app(root).close()
        paths = Paths.resolve(root)
        written = providers_file.load(paths.providers_file)
        written["anthropic"] = ProviderConfig(
            name="anthropic", url="https://api.anthropic.com/v1", api_key="direct-key"
        )
        written["foundry"] = ProviderConfig(
            name="foundry",
            url="https://azure.invalid/anthropic/v1",
            api_key="azure-key",
            deployments={"opus": "claude-opus-5-5"},
        )
        write_atomic(paths.providers_file, providers_file.render(written))
        backend = backend_launch.open_app(root)
        try:
            api = Api(backend)
            before = providers_file.load(paths.providers_file)
            custom = _config(server)
            changed = api.providers.update(
                "generic_anthropic", url=custom.url, api_key=custom.api_key
            )
            assert changed.value.can_edit_url
            listing = api.providers.list(scope="generic_anthropic")
            assert listing.reachable == {"generic_anthropic"}
            assert len(next(p for p in listing.providers if p.answered).models) == 2
            saved = providers_file.load(paths.providers_file)
            for name in ("generic", "anthropic", "foundry"):
                assert saved[name] == before[name]
                assert api.providers.get(name).url == written[name].url
            assert saved["generic_anthropic"].url == custom.url
            assert saved["generic_anthropic"].api_key != custom.api_key
        finally:
            backend.close()
        backend = backend_launch.open_app(root)
        try:
            assert Api(backend).providers.list(scope="generic_anthropic").reachable == {
                "generic_anthropic"
            }
        finally:
            backend.close()


@dataclass(frozen=True)
class _Turn:
    role: str
    body: str
    images: tuple = ()
    volatile: bool = False


def _config(server: ModelServer) -> ProviderConfig:
    server.token_count = 17
    server.api_key = "fake-key"
    server.anthropic_models = [_OPUS, _HAIKU]
    return ProviderConfig(
        name="anthropic",
        url=server.url.removesuffix("/v1") + "/anthropic/v1",
        api_key="fake-key",
    )


def _client(server: ModelServer) -> AnthropicClient:
    return AnthropicClient(_config(server))
