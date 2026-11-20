"""Azure-hosted Claude via Foundry's native Anthropic Messages API.

The wire is Anthropic's (`clients.anthropic`); what Foundry changes is
routing. Deployments are configured explicitly — v2 exposes Messages
and token counting, not a models catalog — each naming the Claude model
it serves, whose known facts (`anthropic.spec_of`) shape its requests.
A listing counts tokens on each deployment to check it without
generating a reply.
"""

from otaku.providers.clients.anthropic import (
    AnthropicClient,
    AnthropicCompletion,
    AnthropicModels,
    Thinking,
    model_info,
    spec_of,
)
from otaku.providers.errors import ProviderError, UnauthorizedError
from otaku.providers.http import Http
from otaku.providers.openai.models import ModelInfo
from otaku.settings.providers import ProviderConfig


class FoundryModels(AnthropicModels):
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
            tokens = count.get("input_tokens") if isinstance(count, dict) else None
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise ProviderError(f"Foundry returned no token count for {deployment}.")
            spec = spec_of(model_id)
            models.append(model_info(deployment, spec))
        return models


class FoundryCompletion(AnthropicCompletion):
    def _validate_model(self, model: str) -> None:
        if model not in self._config.deployments:
            raise ProviderError(f"No Foundry deployment is configured as {model!r}.")

    def _thinking_of(self, model: str) -> Thinking | None:
        spec = spec_of(self._config.deployments.get(model, ""))
        return spec.thinking if spec else None


class FoundryClient(AnthropicClient):
    id = "foundry"
    label = "Azure Foundry Claude"
    env_key = "ANTHROPIC_FOUNDRY_API_KEY"
    models_class = FoundryModels
    completion_class = FoundryCompletion

    @classmethod
    def autoconfigure(cls) -> ProviderConfig:
        # A resource's endpoint is nobody's to guess: never founded.
        return ProviderConfig(name=cls.id, url="")
