"""Tests for the OpenRouter components (no network needed)."""

import json

import httpx
import pytest
from DashAI.back.credentials.base_credential import BaseCredential

from dashai_openrouter import openrouter_credential
from dashai_openrouter.openrouter_credential import OpenRouterCredential
from dashai_openrouter.openrouter_model import OpenRouterModel

PARAMS = {
    "model": "openrouter/free",
    "max_tokens": 64,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 0,
    "reasoning": "auto",
    "timeout": 5,
}
MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture
def server(monkeypatch):
    """Route httpx through an in-memory handler; yields captured requests."""
    seen = []
    state = {"handler": lambda r: httpx.Response(500)}
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return state["handler"](request)

    class Client(real):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(handler), **kw)

    def send(method):
        def call(url, **kw):
            with Client() as c:
                return c.request(method, url, **kw)

        return call

    monkeypatch.setattr(httpx, "post", send("POST"))
    monkeypatch.setattr(httpx, "get", send("GET"))
    BaseCredential.set_key("sk-test")
    return seen, state


def _reply(content="hi", finish="stop"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}, "finish_reason": finish}]},
    )


def test_generate_sends_chat_request_and_returns_reply(server):
    seen, state = server
    state["handler"] = lambda r: _reply()

    assert OpenRouterModel(**PARAMS).generate(MESSAGES) == ["hi"]

    request = seen[0]
    body = json.loads(request.content)
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["x-title"] == "DashAI"
    assert body["messages"] == MESSAGES
    assert body["model"] == PARAMS["model"]
    assert body["max_tokens"] == 64
    assert "top_k" not in body
    assert "reasoning" not in body


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [("off", {"enabled": False}), ("high", {"effort": "high"})],
)
def test_reasoning_is_sent_only_when_chosen(server, reasoning, expected):
    seen, state = server
    state["handler"] = lambda r: _reply()

    OpenRouterModel(**{**PARAMS, "reasoning": reasoning, "top_k": 20}).generate(
        MESSAGES
    )

    body = json.loads(seen[0].content)
    assert body["reasoning"] == expected
    assert body["top_k"] == 20


def test_missing_key_fails_before_any_request(server):
    seen, _ = server
    BaseCredential._keys.clear()

    with pytest.raises(RuntimeError, match="No OpenRouter API key"):
        OpenRouterModel(**PARAMS).generate(MESSAGES)
    assert not seen


@pytest.mark.parametrize(
    ("status", "text"),
    [(401, "invalid or disabled"), (402, "no credits"), (429, "rate limited")],
)
def test_http_errors_explain_what_to_do(server, status, text):
    _, state = server
    state["handler"] = lambda r: httpx.Response(
        status, json={"error": {"code": status, "message": "upstream said no"}}
    )

    with pytest.raises(RuntimeError, match=text):
        OpenRouterModel(**PARAMS).generate(MESSAGES)


def test_error_inside_a_200_is_detected(server):
    _, state = server
    state["handler"] = lambda r: httpx.Response(
        200, json={"error": {"code": 502, "message": "provider failed"}}
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        OpenRouterModel(**PARAMS).generate(MESSAGES)


def test_empty_answer_cut_by_max_tokens_explains_reasoning(server):
    _, state = server
    state["handler"] = lambda r: _reply(content=None, finish="length")

    with pytest.raises(RuntimeError, match="larger max_tokens"):
        OpenRouterModel(**PARAMS).generate(MESSAGES)


def test_unreachable_service_raises_runtime_error(server):
    _, state = server

    def boom(request):
        raise httpx.ConnectError("no route")

    state["handler"] = boom

    with pytest.raises(RuntimeError, match="request failed"):
        OpenRouterModel(**PARAMS).generate(MESSAGES)


def test_credential_verifies_the_key_against_the_service(server):
    seen, state = server
    state["handler"] = lambda r: httpx.Response(
        200 if r.headers["authorization"] == "Bearer good" else 401, json={}
    )

    credential = OpenRouterCredential()

    assert credential.verify("good") is True
    assert credential.verify("bad") is False
    assert seen[0].url == f"{openrouter_credential.API_URL}/key"


def test_model_requires_the_credential():
    assert OpenRouterModel.REQUIRED_CREDENTIALS == ["OpenRouterCredential"]
