"""Foundry deployments through the real provider, app and native HTTP wire."""

import sys
import tomllib
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from otaku.backend import launch as backend_launch
from otaku.backend.api import Api, ModelRef, ModelRole
from otaku.backend.app.paths import Paths
from otaku.encryption import SealedError, is_sealed, unseal
from otaku.encryption.credentials import CredentialError, delete, resolve, store
from otaku.providers import (
    DeclinedError,
    ProviderConfig,
    Reasoning,
    StatusError,
    Text,
    UnreachableError,
)
from otaku.providers.clients.foundry import FoundryClient
from otaku.settings import providers as providers_file
from otaku.settings import write_atomic
from otaku.settings.config import ConfigError
from scenarios.support.harness import App, set_config_provider
from scenarios.support.server import ModelServer
from scripts.configure_foundry import _save_config


class TestDeployment:
    def test_listing_validates_configured_deployments_without_a_models_api(
        self,
        server: ModelServer,
    ) -> None:
        client = _client(server)
        (model,) = client.models.list()
        assert model.name == "custom-opus"
        assert model.context_window == 1_000_000
        assert model.capabilities.vision is True
        assert "none" not in model.capabilities.reasoning_efforts
        assert server.gets == []
        assert server.posts == ["/anthropic/v1/messages/count_tokens"]
        assert server.requests[0]["model"] == "custom-opus"

    def test_streaming_and_counting_use_native_blocks_and_file_the_answer(
        self,
        server: ModelServer,
    ) -> None:
        sink = _Sink()
        client = _client(server, sink)
        server.script = lambda _: ("Thought", "Answer")
        server.cached_tokens = 20
        reply = client.completion.chat("custom-opus", [_Turn("user", "Hi")], {}, level="high")
        chunks = list(reply)
        assert "".join(c.text for c in chunks if isinstance(c, Reasoning)) == "Thought"
        assert "".join(c.text for c in chunks if isinstance(c, Text)) == "Answer"
        assert (
            reply.stats.prompt_tokens,
            reply.stats.completion_tokens,
            reply.stats.cached_tokens,
        ) == (30, 5, 20)
        assert reply.stats.finish_reason == "stop"
        assert server.requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert sink.answers[0]["status"] == "ok"
        assert sink.answers[0]["text"] == "Answer"
        assert "fake-key" not in repr((sink.requests, sink.answers))
        assert client.completion.count_chat_tokens("custom-opus", [_Turn("user", "Hi")]) == 17
        assert server.posts[-1] == "/anthropic/v1/messages/count_tokens"
        assert "stream" not in server.requests[-1]
        assert "max_tokens" not in server.requests[-1]
        assert server.request_headers[0]["x-api-key"] == "fake-key"

    @pytest.mark.parametrize(
        "identity", ["claude-opus-5", "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-5-5"]
    )
    def test_optional_adaptive_thinking_can_be_disabled(
        self, server: ModelServer, identity: str
    ) -> None:
        config = replace(_config(server), deployments={"deployment": identity})
        client = FoundryClient(config)
        (model,) = client.models.list()
        assert "none" in model.capabilities.reasoning_efforts
        list(client.completion.chat("deployment", [_Turn("user", "Hi")], {}, level="none"))
        assert server.requests[-1]["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize("failure", ["truncated", "error", "refusal"])
    def test_incomplete_or_refused_streams_are_not_filed_as_success(
        self,
        server: ModelServer,
        failure: str,
    ) -> None:
        sink = _Sink()
        client = _client(server, sink)
        if failure == "truncated":
            server.no_done = True
        elif failure == "error":
            server.decline = "overloaded"
        else:
            server.finish = "refusal"
        with pytest.raises(UnreachableError if failure == "truncated" else DeclinedError):
            list(client.completion.chat("custom-opus", [_Turn("user", "Hi")], {}))
        assert sink.answers[0]["status"].startswith("failed:")

    def test_closing_a_reply_files_partial_text_as_cancelled(self, server: ModelServer) -> None:
        sink = _Sink()
        client = _client(server, sink)
        server.script = lambda _: "A long answer"
        reply = client.completion.chat("custom-opus", [_Turn("user", "Hi")], {})
        next(reply)
        reply.close()
        assert sink.answers[0]["status"] == "cancelled"
        assert sink.answers[0]["text"]

    def test_an_error_after_partial_text_is_not_retried_or_filed_as_success(
        self, server: ModelServer
    ) -> None:
        sink = _Sink()
        client = _client(server, sink)
        server.script = lambda _: "An incomplete answer"
        server.anthropic_error_after = 1
        received = []
        with pytest.raises(DeclinedError):
            for chunk in client.completion.chat("custom-opus", [_Turn("user", "Hi")], {}):
                if isinstance(chunk, Text):
                    received.append(chunk.text)
        assert received
        assert len(server.posts) == 1
        assert sink.answers[0]["status"].startswith("failed:")

    @pytest.mark.parametrize("error", ["invalid type for message content", "invalid display name"])
    def test_an_unrelated_error_does_not_retry_without_thinking(
        self, server: ModelServer, error: str
    ) -> None:
        client = _client(server)
        server.refuse = lambda _: 400
        server.refusal = error
        with pytest.raises(StatusError):
            list(client.completion.chat("custom-opus", [_Turn("user", "Hi")], {}, level="high"))
        assert len(server.requests) == 1

    def test_a_thinking_refusal_still_retries_without_thinking(self, server: ModelServer) -> None:
        client = _client(server)
        server.refuse = lambda body: 400 if "thinking" in body else None
        server.refusal = "thinking is not supported"
        list(client.completion.chat("custom-opus", [_Turn("user", "Hi")], {}, level="high"))
        assert len(server.requests) == 2
        assert "thinking" in server.requests[0]
        assert "thinking" not in server.requests[1]

    def test_the_real_app_plays_and_persists_a_foundry_reply(
        self,
        tmp_path: Path,
        server: ModelServer,
    ) -> None:
        root = tmp_path / "state"
        set_config_provider(root, server)
        initial = backend_launch.open_app(root)
        initial.close()
        paths = Paths.resolve(root)
        config = _config(server)
        written = providers_file.load(paths.providers_file)
        written["foundry"] = config
        write_atomic(paths.providers_file, providers_file.render(written))
        backend = backend_launch.open_app(root)
        try:
            api = Api(backend)
            api.providers.list(scope="foundry")
            api.app.update(models={ModelRole.WRITING: ModelRef("foundry", "custom-opus")})
            settings = api.providers.models.settings.get("foundry", "custom-opus")
            assert set(settings.choices.levels) == {
                "unset",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }
            assert not settings.choices.budget
            api.providers.models.settings.update("foundry", "custom-opus", think="high")
            story = api.stories.create().value
            events = list(
                api.stories.messages.create(
                    story.id,
                    line="Open the gate.",
                    expected_head=story.head,
                )
            )
            assert events
            messages = api.stories.messages.list(story.id)
            assert messages[-1].body == "The light went out, and something stirred in the dark."
            assert any(p == "/anthropic/v1/messages" for p in server.posts)
            sent = next(body for body in server.requests if body.get("stream") is True)
            assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
            assert sent["output_config"] == {"effort": "high"}
        finally:
            backend.close()


class TestConfiguration:
    @pytest.mark.parametrize(
        ("endpoint", "deployment", "model_id"),
        [
            ("", "opus", "claude-opus-5-5"),
            ("not-a-url", "opus", "claude-opus-5-5"),
            ("https://[invalid", "opus", "claude-opus-5-5"),
            ("https://example/anthropic?version=1", "opus", "claude-opus-5-5"),
            ("https://example/anthropic", " ", "claude-opus-5-5"),
            ("https://example/anthropic", "opus", "not-claude"),
        ],
    )
    def test_invalid_configuration_leaves_the_provider_file_unchanged(
        self, tmp_path: Path, endpoint: str, deployment: str, model_id: str
    ) -> None:
        paths = Paths.resolve(tmp_path)
        paths.ensure_tree()
        paths.providers_file.write_text('[generic]\nurl = "http://localhost/v1"\n')
        before = paths.providers_file.read_bytes()
        with pytest.raises(ConfigError):
            _save_config(paths, endpoint, deployment, model_id, "wincred:otaku/test")
        assert paths.providers_file.read_bytes() == before

    def test_rotating_a_credential_with_the_same_reference_is_already_configured(
        self,
        tmp_path: Path,
    ) -> None:
        paths = Paths.resolve(tmp_path)
        paths.ensure_tree()
        paths.providers_file.write_text(
            '[generic]\nurl = "http://localhost:8080/v1"\napi_key = ""\n',
            encoding="utf-8",
        )
        args = (
            paths,
            "https://example/anthropic/v1/messages",
            "custom-opus",
            "claude-opus-5-5",
            "wincred:otaku/test",
        )
        _save_config(*args)
        before = paths.providers_file.read_bytes()
        _save_config(*args)
        assert paths.providers_file.read_bytes() == before
        assert (
            providers_file.load(paths.providers_file)["generic"].url == "http://localhost:8080/v1"
        )

    def test_a_failed_configuration_write_is_not_reported_as_configured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from otaku.settings.migrations import surgery

        paths = Paths.resolve(tmp_path)
        paths.ensure_tree()
        paths.providers_file.write_text('[generic]\nurl = "http://localhost/v1"\n')
        monkeypatch.setattr(surgery, "update_providers", lambda *args: False)
        with pytest.raises(ConfigError):
            _save_config(
                paths,
                "https://example/anthropic",
                "opus",
                "claude-opus-5-5",
                "wincred:otaku/test",
            )

    def test_deployment_names_round_trip_through_the_provider_file(self, tmp_path: Path) -> None:
        path = tmp_path / "providers.toml"
        config = ProviderConfig(
            name="foundry",
            url="https://example/anthropic",
            deployments={"custom-opus": "claude-opus-5-5"},
        )
        path.write_text(providers_file.render({"foundry": config}), encoding="utf-8")
        assert providers_file.load(path)["foundry"].deployments == config.deployments

    @pytest.mark.parametrize("value", ['["opus"]', "{ opus = 7 }", '{ opus = "gpt" }'])
    def test_invalid_deployment_mappings_are_refused(self, tmp_path: Path, value: str) -> None:
        path = tmp_path / "providers.toml"
        path.write_text(f'[foundry]\nurl = "https://example"\ndeployments = {value}\n')
        with pytest.raises(ConfigError):
            providers_file.load(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Credential Manager")
class TestWindowsCredential:
    @pytest.mark.parametrize("key", ["", "invalid\nkey", "x" * 1281])
    def test_invalid_keys_are_refused_without_storing_a_credential(self, key: str) -> None:
        target = "otaku-test/" + uuid.uuid4().hex
        try:
            with pytest.raises(CredentialError):
                store(target, key)
            with pytest.raises(CredentialError):
                resolve(f"wincred:{target}")
        finally:
            delete(target)

    def test_control_characters_in_a_target_cannot_address_another_credential(self) -> None:
        target = "otaku-test/" + uuid.uuid4().hex
        invalid = target + "\x00suffix"
        try:
            reference = store(target, "original")
            with pytest.raises(CredentialError):
                store(invalid, "replacement")
            with pytest.raises(CredentialError):
                resolve(f"wincred:{invalid}")
            with pytest.raises(CredentialError):
                delete(invalid)
            assert resolve(reference) == "original"
        finally:
            delete(target)

    def test_reference_resolves_without_a_sealing_key_file_and_a_missing_item_is_refused(
        self,
        tmp_path: Path,
        app: App,
    ) -> None:
        target = "otaku-test/" + uuid.uuid4().hex
        reference = store(target, "fake-secret")
        try:
            assert is_sealed(reference)
            assert resolve(reference) == "fake-secret"
            assert unseal(reference, key_file=tmp_path / "absent", service="test") == "fake-secret"
            app.api.providers.update("generic", api_key=reference)
            raw = app.paths.providers_file.read_text(encoding="utf-8")
            assert "fake-secret" not in raw
            assert tomllib.loads(raw)["generic"]["api_key"] == reference
            assert not (tmp_path / "absent").exists()
            delete(target)
            with pytest.raises(CredentialError):
                resolve(reference)
            with pytest.raises(SealedError):
                unseal(reference, key_file=tmp_path / "absent", service="test")
        finally:
            delete(target)


@dataclass(frozen=True)
class _Turn:
    role: str
    body: str
    images: tuple = ()
    volatile: bool = False


@dataclass
class _Sink:
    requests: list[dict[str, object]] = field(default_factory=list)
    answers: list[dict[str, object]] = field(default_factory=list)

    def record_request(self, provider: str, purpose: str, body: dict[str, object]) -> str:
        self.requests.append(body)
        return str(len(self.requests))

    def record_answer(self, provider: str, purpose: str, request_id: str, **kwargs: object) -> None:
        self.answers.append(kwargs)


def _config(server: ModelServer) -> ProviderConfig:
    server.token_count = 17
    server.api_key = "fake-key"
    return ProviderConfig(
        name="foundry",
        url=server.url.removesuffix("/v1") + "/anthropic/v1",
        api_key="fake-key",
        deployments={"custom-opus": "claude-opus-5-5"},
    )


def _client(server: ModelServer, sink: _Sink | None = None) -> FoundryClient:
    return FoundryClient(_config(server), request_sink=sink)
