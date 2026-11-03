"""Smoke test against a real DashAI install.

The regular tests run against ``stubs/``, which can drift from the host. This
file catches that drift: it needs the real DashAI, so it is deselected by
default and run by the scheduled workflow:

    pytest -m dashai -o pythonpath=src -o addopts=""
"""

import pathlib

import pytest

pytestmark = pytest.mark.dashai

COMPONENTS = ["OpenRouterCredential", "OpenRouterModel"]


@pytest.fixture(scope="module")
def registry():
    import DashAI

    if "stubs" in pathlib.Path(DashAI.__file__).parts:
        pytest.fail("This test needs the real DashAI, but it imported stubs/.")

    from DashAI.back.dependencies.registry.component_registry import ComponentRegistry

    from dashai_openrouter.openrouter_credential import OpenRouterCredential
    from dashai_openrouter.openrouter_model import OpenRouterModel

    components = ComponentRegistry()
    for component in (OpenRouterCredential, OpenRouterModel):
        components.register_component(component)
    return components


@pytest.mark.parametrize("name", COMPONENTS)
def test_component_registers(registry, name):
    assert name in registry


def test_credential_is_listed_as_a_credential(registry):
    assert "OpenRouterCredential" in registry._registry["Credential"]


def test_model_works_for_chat_and_rag_and_needs_the_credential(registry):
    component = registry["OpenRouterModel"]["class"]

    assert {"TextToTextGenerationTask", "RAGTask"} <= set(
        component.COMPATIBLE_COMPONENTS
    )
    assert registry["OpenRouterModel"]["required_credentials"] == [
        "OpenRouterCredential"
    ]


def test_default_parameters_build_the_model(registry):
    component = registry["OpenRouterModel"]["class"]
    properties = component.get_schema()["properties"]
    defaults = {key: spec["placeholder"] for key, spec in properties.items()}

    # Building the model makes no network call.
    component(**defaults)
