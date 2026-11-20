"""Anthropic's own API through the real provider, app and native HTTP wire."""

from dataclasses import dataclass
from pathlib import Path

from otaku.backend import launch as backend_launch
from otaku.backend.api import Api, ModelRef, ModelRole
from otaku.backend.app.paths import Paths
from otaku.providers import ProviderConfig, Reasoning, Text, autoconfigure
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
        assert sent["thinking"] == {"type": "adaptive"}
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


class TestSetup:
    def test_the_api_key_variable_founds_the_section(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        founded = autoconfigure()
        assert founded["anthropic"].url == "https://api.anthropic.com/v1"
        assert "foundry" not in founded

    def test_the_real_app_plays_and_persists_an_anthropic_reply(
        self, tmp_path: Path, server: ModelServer
    ) -> None:
        root = tmp_path / "state"
        set_config_provider(root, server)
        backend_launch.open_app(root).close()
        paths = Paths.resolve(root)
        written = providers_file.load(paths.providers_file)
        written["anthropic"] = _config(server)
        write_atomic(paths.providers_file, providers_file.render(written))
        backend = backend_launch.open_app(root)
        try:
            api = Api(backend)
            api.providers.list(scope="anthropic")
            api.app.update(models={ModelRole.WRITING: ModelRef("anthropic", "claude-opus-5-5")})
            story = api.stories.create().value
            assert list(
                api.stories.messages.create(
                    story.id, line="Open the gate.", expected_head=story.head
                )
            )
            messages = api.stories.messages.list(story.id)
            assert messages[-1].body == "The light went out, and something stirred in the dark."
            assert "/anthropic/v1/messages" in server.posts
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
