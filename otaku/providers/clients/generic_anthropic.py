"""A custom Anthropic-compatible endpoint, independent of Anthropic and
Foundry. The Messages wire and model discovery are Anthropic's; the url
and key are the reader's, and the server may be local or hosted. No
endpoint is assumed and no first-run section is written.
"""

from otaku.providers.clients.anthropic import AnthropicClient
from otaku.providers.openai.models import Locality
from otaku.settings.providers import ProviderConfig


class GenericAnthropicClient(AnthropicClient):
    id = "generic_anthropic"
    label = "Generic Anthropic provider"
    locality = Locality.UNKNOWN
    env_key = "GENERIC_ANTHROPIC_API_KEY"

    @classmethod
    def autoconfigure(cls) -> ProviderConfig:
        return ProviderConfig(name=cls.id, url="")
