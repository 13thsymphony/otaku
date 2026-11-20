"""Configure a Foundry deployment using a local masked API-key prompt."""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

from otaku.backend.app.paths import Paths
from otaku.encryption.credentials import CredentialError, store
from otaku.formatting import toml_key, toml_scalar
from otaku.settings.config import ConfigError
from otaku.settings.migrations import surgery
from otaku.settings.providers import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--target", default="otaku/azure-foundry")
    parser.add_argument(
        "--home",
        type=Path,
        required=True,
        help="An existing otaku home folder (or a new isolated test folder).",
    )
    args = parser.parse_args()
    if sys.platform != "win32" or not sys.stdin.isatty():
        parser.error("Run in an interactive Windows terminal; keys are never accepted on stdin.")
    paths = Paths.resolve(args.home)
    if not paths.providers_file.exists():
        parser.error("Initialize this home with otaku first, then run this command.")
    # Validate before asking for or storing a key.
    try:
        load(paths.providers_file)
        _validate_config(args.endpoint, args.deployment, args.model)
    except ConfigError as e:
        parser.error(str(e))
    try:
        reference = store(args.target, _read_key())
    except CredentialError as e:
        parser.exit(1, f"Credential not stored: {e}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nCancelled; credential not stored.\n")
    try:
        _save_config(paths, args.endpoint, args.deployment, args.model, reference)
    except ConfigError as e:
        parser.exit(1, f"Credential stored, but configuration not saved: {e}\n")
    print("Foundry configured. Restart otaku and select the deployment in the model picker.")


def _save_config(
    paths: Paths,
    endpoint: str,
    deployment: str,
    model_id: str,
    reference: str,
) -> None:
    _validate_config(endpoint, deployment, model_id)
    config = load(paths.providers_file).get("foundry")
    deployments = dict(config.deployments) if config else {}
    deployments[deployment] = model_id
    entries = ", ".join(
        f"{toml_key(name)} = {toml_scalar(model)}" for name, model in deployments.items()
    )
    edits = [
        surgery.ensure_section("foundry", '[foundry]\nurl = ""\napi_key = ""'),
        surgery.set_key("foundry", "url", f"url = {toml_scalar(endpoint)}"),
        surgery.set_key("foundry", "api_key", f"api_key = {toml_scalar(reference)}"),
        surgery.set_key("foundry", "deployments", f"deployments = {{ {entries} }}"),
        surgery.ensure_key("foundry", "prompt_cache", 'prompt_cache = "5m"'),
    ]
    surgery.update_providers(
        paths.providers_file,
        paths.config_backups_dir,
        edits,
    )
    saved = load(paths.providers_file).get("foundry")
    if (
        saved is None
        or saved.url != endpoint.rstrip("/")
        or saved.api_key != reference
        or saved.deployments != deployments
    ):
        raise ConfigError("providers.toml could not be updated; run again to retry.")


def _validate_config(endpoint: str, deployment: str, model_id: str) -> None:
    try:
        url = urlsplit(endpoint)
        is_valid_endpoint = bool(
            endpoint == endpoint.strip()
            and url.scheme in ("http", "https")
            and url.hostname
            and not url.query
            and not url.fragment
        )
    except ValueError:
        is_valid_endpoint = False
    if not is_valid_endpoint:
        raise ConfigError(
            "The Foundry endpoint must be an HTTP(S) URL without a query or fragment."
        )
    if not deployment.strip():
        raise ConfigError("The Foundry deployment name must not be empty.")
    if not model_id.startswith("claude-"):
        raise ConfigError("The Foundry model id must start with 'claude-'.")


def _read_key() -> str:
    """Read without console echo, showing only stars; reject control input."""
    if sys.platform != "win32":
        raise CredentialError("A Windows terminal is required to read the key.")
    import msvcrt

    # PowerShell leaves bracketed paste on; raw secret input must not
    # receive terminal framing as part of the credential.
    sys.stdout.write("\x1b[?2004l")
    print("Right-click to paste. Input is masked; Backspace edits; Ctrl+C cancels.")
    print("Foundry API key: ", end="", flush=True)
    chars: list[str] = []
    while True:
        char = msvcrt.getwch()
        if char in ("\r", "\n"):
            print()
            return "".join(chars).strip()
        if char in ("\x03", "\x04", "\x1a"):
            raise KeyboardInterrupt
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()  # consume an arrow or function key's scan code
            continue
        if char in ("\b", "\x7f"):
            if chars:
                chars.pop()
                print("\b \b", end="", flush=True)
            continue
        if char == "\x16":
            raise CredentialError(
                "Ctrl+V was received as a key. Run again and right-click to paste."
            )
        if not char.isprintable():
            raise CredentialError("Invalid control input. Run again and right-click to paste.")
        chars.append(char)
        print("*", end="", flush=True)


if __name__ == "__main__":
    main()
