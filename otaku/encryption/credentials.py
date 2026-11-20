"""API-key references backed by Windows Credential Manager.

Only a wincred:target reference belongs in configuration. The generic
credential's blob is protected by Windows for the current user; keys
never ride on command arguments or fall back to a file.
"""

import ctypes
import sys
from ctypes import wintypes
from typing import Any

PREFIX = "wincred:"
_GENERIC = 1
_LOCAL_MACHINE = 2
_NOT_FOUND = 1168


class CredentialError(Exception):
    pass


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def store(target: str, api_key: str) -> str:
    """Save a generic credential for the current user; return its reference."""
    if not target or not api_key:
        raise CredentialError("A credential target and a nonempty API key are required.")
    _validate_target(target)
    if any(ord(c) < 32 or ord(c) == 127 for c in api_key):
        raise CredentialError(
            "The API key contains control characters; enter it again at the secure prompt."
        )
    api = _api()
    encoded = api_key.encode("utf-16-le")
    if len(encoded) > 2560:
        raise CredentialError("The API key exceeds Windows Credential Manager's blob limit.")
    blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    credential = _Credential()
    credential.Type = _GENERIC
    credential.TargetName = target
    credential.UserName = "otaku"
    credential.CredentialBlobSize = len(encoded)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = _LOCAL_MACHINE
    try:
        if not api.CredWriteW(ctypes.byref(credential), 0):
            raise _failure("store")
    finally:
        ctypes.memset(blob, 0, len(encoded))
    return PREFIX + target


def resolve(reference: str) -> str:
    """Read one reference; a missing credential is an explicit error."""
    if not reference.startswith(PREFIX) or not reference[len(PREFIX) :]:
        raise CredentialError("Invalid Windows credential reference.")
    target = reference[len(PREFIX) :]
    _validate_target(target)
    api = _api()
    found = ctypes.POINTER(_Credential)()
    if not api.CredReadW(target, _GENERIC, 0, ctypes.byref(found)):
        raise _failure("read")
    try:
        blob = ctypes.string_at(found.contents.CredentialBlob, found.contents.CredentialBlobSize)
        try:
            key = blob.decode("utf-16-le")
        except UnicodeDecodeError as e:
            raise CredentialError("The Windows credential is not a valid API key.") from e
        if not key:
            raise CredentialError("The Windows credential contains an empty API key.")
        return key
    finally:
        api.CredFree(found)


def delete(target: str) -> None:
    """Remove exactly one generic credential; absence is already removed."""
    _validate_target(target)
    api = _api()
    if not api.CredDeleteW(target, _GENERIC, 0) and _last_error() != _NOT_FOUND:
        raise _failure("delete")


def _validate_target(target: str) -> None:
    if not target or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in target):
        raise CredentialError(
            "A credential target must be nonempty and contain no control characters."
        )


def _api() -> Any:
    if sys.platform != "win32":
        raise CredentialError("Windows credential references require Windows Credential Manager.")
    api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    api.CredWriteW.argtypes = [ctypes.POINTER(_Credential), wintypes.DWORD]
    api.CredWriteW.restype = wintypes.BOOL
    api.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_Credential)),
    ]
    api.CredReadW.restype = wintypes.BOOL
    api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    api.CredDeleteW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    api.CredFree.restype = None
    return api


def _failure(operation: str) -> CredentialError:
    code = _last_error()
    return CredentialError(f"Windows credential {operation} failed (error {code}).")


def _last_error() -> int:
    if sys.platform != "win32":
        raise CredentialError("Windows credential references require Windows Credential Manager.")
    return ctypes.get_last_error()
