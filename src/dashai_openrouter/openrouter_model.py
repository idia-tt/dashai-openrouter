"""Text generation through OpenRouter."""

from typing import ClassVar

import httpx
from DashAI.back.core.schema_fields import (
    BaseSchema,
    enum_field,
    float_field,
    int_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.text_to_text_generation_model import (
    TextToTextGenerationTaskModel,
)

from dashai_openrouter.openrouter_credential import API_URL, auth_headers

DEFAULT_MODEL = "openrouter/free"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_REASONING = "auto"
DEFAULT_TIMEOUT = 120

# What each failure means for the user, from the OpenRouter error reference.
_STATUS_HINTS = {
    400: "the request was rejected, check the model name and parameters",
    401: "the API key is invalid or disabled, set it again in the credentials",
    402: "the account or the key has no credits left",
    403: "the request was blocked (permissions, moderation or a guardrail)",
    408: "the request timed out",
    429: "rate limited, wait a moment and try again",
    502: "the model is down or returned an invalid response",
    503: "no provider can serve the request with these settings",
}


class OpenRouterSchema(BaseSchema):
    """Hyperparameters of a model served through OpenRouter."""

    model: schema_field(
        string_field(),
        placeholder=DEFAULT_MODEL,
        description=MultilingualString(
            en=(
                "Model id as shown in the OpenRouter catalog, in the form "
                "`provider/model` (for example `anthropic/claude-sonnet-4.5`). "
                "Ids ending in `:free` cost nothing. Browse the "
                "[catalog](https://openrouter.ai/models)."
            ),
            es=(
                "Id del modelo tal como aparece en el catálogo de OpenRouter, con "
                "la forma `proveedor/modelo` (por ejemplo "
                "`anthropic/claude-sonnet-4.5`). Los ids que terminan en `:free` "
                "no tienen costo. Explore el "
                "[catálogo](https://openrouter.ai/models)."
            ),
            pt=(
                "Id do modelo como aparece no catálogo do OpenRouter, na forma "
                "`provedor/modelo` (por exemplo `anthropic/claude-sonnet-4.5`). "
                "Ids terminados em `:free` não têm custo. Veja o "
                "[catálogo](https://openrouter.ai/models)."
            ),
            de=(
                "Modell-ID wie im OpenRouter-Katalog, in der Form "
                "`anbieter/modell` (z. B. `anthropic/claude-sonnet-4.5`). IDs mit "
                "der Endung `:free` sind kostenlos. Siehe den "
                "[Katalog](https://openrouter.ai/models)."
            ),
            zh=(
                "OpenRouter 目录中显示的模型 ID，格式为 `提供商/模型`（例如 "
                "`anthropic/claude-sonnet-4.5`）。以 `:free` 结尾的 ID 免费。"
                "浏览[目录](https://openrouter.ai/models)。"
            ),
        ),
        alias=MultilingualString(
            en="Model",
            es="Modelo",
            pt="Modelo",
            de="Modell",
            zh="模型",
        ),
    )  # ty: ignore[invalid-type-form]

    max_tokens: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_MAX_TOKENS,
        description=MultilingualString(
            en=(
                "Maximum number of new tokens per response. About 1 token is 0.75 "
                "English words. Reasoning models also spend tokens on thinking, so "
                "give them 4096 or more or the answer is cut off. You pay for "
                "the tokens generated."
            ),
            es=(
                "Número máximo de tokens nuevos por respuesta. Aproximadamente 1 "
                "token son 0,75 palabras en inglés. Los modelos de razonamiento "
                "también gastan tokens pensando: déles 4096 o más o la respuesta "
                "se corta. Se paga por los tokens generados."
            ),
            pt=(
                "Número máximo de novos tokens por resposta. Cerca de 1 token "
                "equivale a 0,75 palavras em inglês. Modelos de raciocínio também "
                "gastam tokens pensando: dê 4096 ou mais ou a resposta é cortada. "
                "Paga-se pelos tokens gerados."
            ),
            de=(
                "Maximale Anzahl neuer Token pro Antwort. Etwa 1 Token entspricht "
                "0,75 englischen Wörtern. Reasoning-Modelle verbrauchen auch Token "
                "fürs Denken: 4096 oder mehr geben, sonst wird die Antwort "
                "abgeschnitten. Bezahlt werden die erzeugten Token."
            ),
            zh=(
                "每次回复的最大新 token 数。约 1 个 token 相当于 0.75 个英文单词。"
                "推理模型还会在思考上消耗 token，请设为 4096 或更多，否则回答会被截断。"
                "按生成的 token 计费。"
            ),
        ),
        alias=MultilingualString(
            en="Max tokens",
            es="Tokens máximos",
            pt="Tokens máximos",
            de="Maximale neue Token",
            zh="最大 token 数",
        ),
    )  # ty: ignore[invalid-type-form]

    temperature: schema_field(
        float_field(ge=0.0, le=2.0),
        placeholder=DEFAULT_TEMPERATURE,
        description=MultilingualString(
            en=(
                "Randomness of the output. Low values (0.1-0.3) give focused, "
                "repeatable answers; high values (1.0 or more) give more varied "
                "ones."
            ),
            es=(
                "Aleatoriedad de la salida. Valores bajos (0,1-0,3) dan respuestas "
                "enfocadas y repetibles; valores altos (1,0 o más) dan respuestas "
                "más variadas."
            ),
            pt=(
                "Aleatoriedade da saída. Valores baixos (0,1-0,3) dão respostas "
                "focadas e repetíveis; valores altos (1,0 ou mais) dão respostas "
                "mais variadas."
            ),
            de=(
                "Zufälligkeit der Ausgabe. Niedrige Werte (0,1-0,3) liefern "
                "fokussierte, wiederholbare Antworten; hohe Werte (1,0 oder mehr) "
                "abwechslungsreichere."
            ),
            zh=(
                "输出的随机性。较低的值（0.1-0.3）给出集中、可重复的回答；"
                "较高的值（1.0 以上）给出更多样的回答。"
            ),
        ),
        alias=MultilingualString(
            en="Temperature",
            es="Temperatura",
            pt="Temperatura",
            de="Temperatur",
            zh="温度",
        ),
    )  # ty: ignore[invalid-type-form]

    top_p: schema_field(
        float_field(ge=0.0, le=1.0),
        placeholder=DEFAULT_TOP_P,
        description=MultilingualString(
            en=(
                "Nucleus sampling: only the most likely tokens whose probabilities "
                "add up to this value are considered. 1.0 considers them all."
            ),
            es=(
                "Muestreo por núcleo: solo se consideran los tokens más probables "
                "cuyas probabilidades suman este valor. 1,0 los considera todos."
            ),
            pt=(
                "Amostragem por núcleo: só os tokens mais prováveis cujas "
                "probabilidades somam este valor são considerados. 1,0 considera "
                "todos."
            ),
            de=(
                "Nucleus-Sampling: Es werden nur die wahrscheinlichsten Token "
                "berücksichtigt, deren Wahrscheinlichkeiten diesen Wert ergeben. "
                "1,0 berücksichtigt alle."
            ),
            zh="核采样：只考虑累计概率达到该值的最可能的 token。1.0 表示全部考虑。",
        ),
        alias=MultilingualString(
            en="Top P",
            es="Top P",
            pt="Top P",
            de="Top P",
            zh="Top P",
        ),
    )  # ty: ignore[invalid-type-form]

    top_k: schema_field(
        int_field(ge=0),
        placeholder=DEFAULT_TOP_K,
        description=MultilingualString(
            en=(
                "Only the K most likely tokens are considered at each step. 0 "
                "turns it off. Not every model supports it."
            ),
            es=(
                "Solo se consideran los K tokens más probables en cada paso. 0 lo "
                "desactiva. No todos los modelos lo admiten."
            ),
            pt=(
                "Só os K tokens mais prováveis são considerados em cada passo. 0 "
                "desativa. Nem todos os modelos o suportam."
            ),
            de=(
                "Pro Schritt werden nur die K wahrscheinlichsten Token "
                "berücksichtigt. 0 schaltet es aus. Nicht jedes Modell "
                "unterstützt es."
            ),
            zh="每一步只考虑概率最高的 K 个 token。0 表示关闭。并非所有模型都支持。",
        ),
        alias=MultilingualString(
            en="Top K",
            es="Top K",
            pt="Top K",
            de="Top K",
            zh="Top K",
        ),
    )  # ty: ignore[invalid-type-form]

    reasoning: schema_field(
        enum_field(
            enum=["auto", "off", "low", "medium", "high"],
            labels={
                "auto": MultilingualString(
                    en="Auto",
                    es="Automático",
                    pt="Automático",
                    de="Automatisch",
                    zh="自动",
                ),
                "off": MultilingualString(
                    en="Off",
                    es="Desactivado",
                    pt="Desativado",
                    de="Aus",
                    zh="关闭",
                ),
                "low": MultilingualString(
                    en="Low",
                    es="Bajo",
                    pt="Baixo",
                    de="Niedrig",
                    zh="低",
                ),
                "medium": MultilingualString(
                    en="Medium",
                    es="Medio",
                    pt="Médio",
                    de="Mittel",
                    zh="中",
                ),
                "high": MultilingualString(
                    en="High",
                    es="Alto",
                    pt="Alto",
                    de="Hoch",
                    zh="高",
                ),
            },
        ),
        placeholder=DEFAULT_REASONING,
        description=MultilingualString(
            en=(
                "How much the model thinks before answering. *Auto* keeps the "
                "model's own default. *Off* is rejected by models that always "
                "reason. Thinking tokens count against Max tokens and are billed."
            ),
            es=(
                "Cuánto piensa el modelo antes de responder. *Automático* conserva "
                "el valor propio del modelo. *Desactivado* lo rechazan los modelos "
                "que siempre razonan. Los tokens de razonamiento cuentan para "
                "Tokens máximos y se cobran."
            ),
            pt=(
                "Quanto o modelo pensa antes de responder. *Automático* mantém o "
                "padrão do próprio modelo. *Desativado* é rejeitado por modelos "
                "que sempre raciocinam. Os tokens de raciocínio contam para Tokens "
                "máximos e são cobrados."
            ),
            de=(
                "Wie viel das Modell vor der Antwort nachdenkt. *Automatisch* "
                "behält den Standard des Modells. *Aus* lehnen Modelle ab, die "
                "immer denken. Denk-Token zählen zu Maximale neue Token und "
                "werden berechnet."
            ),
            zh=(
                "模型在回答前思考的程度。*自动*保留模型自身的默认值。始终推理的模型"
                "会拒绝*关闭*。思考 token 计入最大 token 数并计费。"
            ),
        ),
        alias=MultilingualString(
            en="Reasoning",
            es="Razonamiento",
            pt="Raciocínio",
            de="Denkmodus",
            zh="推理",
        ),
    )  # ty: ignore[invalid-type-form]

    timeout: schema_field(
        int_field(ge=1),
        placeholder=DEFAULT_TIMEOUT,
        description=MultilingualString(
            en="Seconds to wait for a response before giving up.",
            es="Segundos de espera por una respuesta antes de rendirse.",
            pt="Segundos de espera por uma resposta antes de desistir.",
            de="Sekunden, die auf eine Antwort gewartet wird, bevor abgebrochen wird.",
            zh="等待响应的秒数，超时则放弃。",
        ),
        alias=MultilingualString(
            en="Timeout (s)",
            es="Tiempo límite (s)",
            pt="Tempo limite (s)",
            de="Zeitlimit (s)",
            zh="超时（秒）",
        ),
    )  # ty: ignore[invalid-type-form]


class OpenRouterModel(TextToTextGenerationTaskModel):
    """Chat model served through OpenRouter.

    Needs an OpenRouter API key, stored with the ``OpenRouterCredential``
    component of this plugin. Requires ``pip install dashai-openrouter``.
    """

    SCHEMA = OpenRouterSchema
    REQUIRED_CREDENTIALS: ClassVar[list[str]] = ["OpenRouterCredential"]
    COLOR: str = "#6467f2"
    DISPLAY_NAME = MultilingualString(
        en="OpenRouter Model",
        es="Modelo OpenRouter",
        pt="Modelo OpenRouter",
        de="OpenRouter-Modell",
        zh="OpenRouter 模型",
    )
    DESCRIPTION = MultilingualString(
        en=(
            "Chat with hundreds of hosted models through one OpenRouter account, "
            "or use one as the generation model of a RAG pipeline. Setup: 1) "
            "Create an API key at [openrouter.ai/keys](https://openrouter.ai/keys) "
            "and save it in the OpenRouter credential. 2) Pick a model from the "
            "[catalog](https://openrouter.ai/models) and type its id in the Model "
            "field. Models ending in `:free` cost nothing, the others are billed "
            "to your account. Your messages are sent to OpenRouter and the "
            "provider of the model. Source code and documentation: "
            "[dashai-openrouter on "
            "GitHub](https://github.com/idia-tt/dashai-openrouter)."
        ),
        es=(
            "Chatee con cientos de modelos alojados a través de una sola cuenta "
            "de OpenRouter, o use uno como modelo de generación de un pipeline "
            "RAG. Configuración: 1) Cree una clave de API en "
            "[openrouter.ai/keys](https://openrouter.ai/keys) y guárdela en la "
            "credencial de OpenRouter. 2) Elija un modelo del "
            "[catálogo](https://openrouter.ai/models) y escriba su id en el "
            "campo Modelo. Los que terminan en `:free` no tienen costo; los demás "
            "se cobran a su cuenta. Sus mensajes se envían a OpenRouter y al "
            "proveedor del modelo. Código fuente y documentación: "
            "[dashai-openrouter en "
            "GitHub](https://github.com/idia-tt/dashai-openrouter)."
        ),
        pt=(
            "Converse com centenas de modelos hospedados por meio de uma única "
            "conta do OpenRouter, ou use um como modelo de geração de um pipeline "
            "RAG. Configuração: 1) Crie uma chave de API em "
            "[openrouter.ai/keys](https://openrouter.ai/keys) e salve-a na "
            "credencial do OpenRouter. 2) Escolha um modelo do "
            "[catálogo](https://openrouter.ai/models) e digite seu id no campo "
            "Modelo. Os terminados em `:free` não têm custo; os demais são "
            "cobrados na sua conta. Suas mensagens são enviadas ao OpenRouter e "
            "ao provedor do modelo. Código-fonte e documentação: "
            "[dashai-openrouter no "
            "GitHub](https://github.com/idia-tt/dashai-openrouter)."
        ),
        de=(
            "Chatten Sie über ein einziges OpenRouter-Konto mit Hunderten "
            "gehosteter Modelle oder nutzen Sie eines als Generierungsmodell "
            "einer RAG-Pipeline. Einrichtung: 1) Einen API-Schlüssel unter "
            "[openrouter.ai/keys](https://openrouter.ai/keys) erstellen und in "
            "der OpenRouter-Zugangsdaten speichern. 2) Ein Modell aus dem "
            "[Katalog](https://openrouter.ai/models) wählen und seine ID im Feld "
            "Modell eintragen. Modelle mit der Endung `:free` sind kostenlos, "
            "die anderen werden Ihrem Konto berechnet. Ihre Nachrichten gehen an "
            "OpenRouter und den Anbieter des Modells. Quellcode und "
            "Dokumentation: [dashai-openrouter auf "
            "GitHub](https://github.com/idia-tt/dashai-openrouter)."
        ),
        zh=(
            "通过一个 OpenRouter 账户与数百个托管模型聊天，或将其中一个用作 RAG "
            "流水线的生成模型。设置步骤：1）在 "
            "[openrouter.ai/keys](https://openrouter.ai/keys) 创建 API 密钥，并保存到"
            " OpenRouter 凭据中。2）从[目录](https://openrouter.ai/models)选择模型，"
            "在“模型”字段中输入其 ID。以 `:free` 结尾的模型免费，其余按账户计费。"
            "您的消息会发送给 OpenRouter 及该模型的提供商。源代码与文档："
            "[GitHub 上的 dashai-openrouter]"
            "(https://github.com/idia-tt/dashai-openrouter)。"
        ),
    )

    def __init__(self, **kwargs):
        """Store the request settings; no network call is made here."""
        kwargs = self.validate_and_transform(kwargs)
        self.model_name = kwargs["model"]
        self.timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
        self.body = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
        }
        # 0 means off, and "auto" sends nothing so each model keeps its default.
        if kwargs.get("top_k", DEFAULT_TOP_K) > 0:
            self.body["top_k"] = kwargs["top_k"]
        reasoning = kwargs.get("reasoning", DEFAULT_REASONING)
        if reasoning == "off":
            self.body["reasoning"] = {"enabled": False}
        elif reasoning != "auto":
            self.body["reasoning"] = {"effort": reasoning}

    def generate(self, prompt: list) -> list[str]:
        """Generate a reply for a conversation.

        Parameters
        ----------
        prompt : list of dict
            Conversation in chat format, each dict with ``role`` and
            ``content``.

        Returns
        -------
        list of str
            A single-element list with the model's reply.

        Raises
        ------
        RuntimeError
            If there is no API key, the service is unreachable, or OpenRouter
            reports an error.
        """
        key = self.get_credential("OpenRouterCredential").get_key()
        if not key:
            raise RuntimeError(
                "No OpenRouter API key is saved. Add it in the credentials."
            )
        try:
            response = httpx.post(
                f"{API_URL}/chat/completions",
                headers=auth_headers(key),
                json={**self.body, "messages": prompt},
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"OpenRouter request failed for model '{self.model_name}': "
                f"{e!r}. Is there an internet connection?"
            ) from e
        try:
            data = response.json()
        except ValueError:
            data = {}
        # OpenRouter can answer 200 with only an error in the body.
        error = data.get("error") if isinstance(data, dict) else None
        if response.status_code != 200 or error or not data.get("choices"):
            detail = (error or {}).get("message") or response.text[:200]
            hint = _STATUS_HINTS.get(response.status_code, "unexpected response")
            raise RuntimeError(
                f"OpenRouter error {response.status_code} for model "
                f"'{self.model_name}': {hint}. {detail}"
            )
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        if not content and choice.get("finish_reason") == "length":
            # Reasoning models spend max_tokens on thinking first.
            raise RuntimeError(
                f"Model '{self.model_name}' hit max_tokens "
                f"({self.body['max_tokens']}) before producing an answer. "
                "Reasoning models need a larger max_tokens."
            )
        return [content]
