"""API key of an OpenRouter account."""

import httpx
from DashAI.back.core.utils import MultilingualString
from DashAI.back.credentials.base_credential import BaseCredential

API_URL = "https://openrouter.ai/api/v1"


def auth_headers(key: str) -> dict[str, str]:
    """Headers every OpenRouter request carries: the key and the app attribution."""
    return {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/idia-tt/dashai-openrouter",
        "X-Title": "DashAI",
    }


class OpenRouterCredential(BaseCredential):
    """API key for OpenRouter, stored encrypted by DashAI."""

    DISPLAY_NAME = MultilingualString(
        en="OpenRouter",
        es="OpenRouter",
        pt="OpenRouter",
        de="OpenRouter",
        zh="OpenRouter",
    )
    DESCRIPTION = MultilingualString(
        en=(
            "API key of your OpenRouter account. Create one at "
            "[openrouter.ai/keys](https://openrouter.ai/keys). Required by the "
            "OpenRouter models."
        ),
        es=(
            "Clave de API de su cuenta de OpenRouter. Cree una en "
            "[openrouter.ai/keys](https://openrouter.ai/keys). La necesitan los "
            "modelos de OpenRouter."
        ),
        pt=(
            "Chave de API da sua conta do OpenRouter. Crie uma em "
            "[openrouter.ai/keys](https://openrouter.ai/keys). Necessária para os "
            "modelos do OpenRouter."
        ),
        de=(
            "API-Schlüssel Ihres OpenRouter-Kontos. Erstellen Sie einen unter "
            "[openrouter.ai/keys](https://openrouter.ai/keys). Erforderlich für "
            "die OpenRouter-Modelle."
        ),
        zh=(
            "您的 OpenRouter 账户的 API 密钥。可在 "
            "[openrouter.ai/keys](https://openrouter.ai/keys) 创建。"
            "OpenRouter 模型需要它。"
        ),
    )
    ICON: str = "Key"

    def verify(self, key: str) -> bool:
        """Check the key against ``GET /key``, which needs a valid key."""
        try:
            response = httpx.get(
                f"{API_URL}/key", headers=auth_headers(key), timeout=15
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200
