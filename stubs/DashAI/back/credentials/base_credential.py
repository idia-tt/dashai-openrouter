"""Stand-in for ``DashAI.back.credentials.base_credential``."""

from typing import Any, ClassVar


class BaseCredential:
    """Keeps the key in memory; the real one is encrypted in the host database."""

    _keys: ClassVar[dict[str, str]] = {}

    def get_key(self) -> str | None:
        return self._keys.get(type(self).__name__)

    @classmethod
    def set_key(cls, key: str) -> None:
        BaseCredential._keys[cls.__name__] = key

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        return {}
