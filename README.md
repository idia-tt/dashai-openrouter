# dashai-openrouter

[![CI](https://github.com/idia-tt/dashai-openrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/idia-tt/dashai-openrouter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [DashAI](https://github.com/DashAISoftware/DashAI) plugin that runs **chat** and **RAG
generation** on hundreds of hosted models through one [OpenRouter](https://openrouter.ai)
account.

| Component             | Shown in DashAI as     | Use it for                                                  |
| --------------------- | ---------------------- | ----------------------------------------------------------- |
| `OpenRouterCredential`| **OpenRouter**         | Your API key, stored encrypted by DashAI                    |
| `OpenRouterModel`     | **OpenRouter Model**   | Chat sessions and the generation model of a RAG pipeline    |

## Quickstart

1. Create an API key at <https://openrouter.ai/keys>.
2. Install DashAI with the plugin and start it:

   ```bash
   uv tool install dashai --with dashai-openrouter
   dashai
   ```

3. Open <http://localhost:8000>, save the key in the **OpenRouter** credential, go to
   **Generative**, create a session and pick **OpenRouter Model**.

The default model, `openrouter/free`, costs nothing. Models without the `:free`
suffix are billed to your OpenRouter account.

## Parameters

| Parameter    | Default                    | Notes                                                                        |
| ------------ | -------------------------- | ---------------------------------------------------------------------------- |
| Model        | `openrouter/free`  | Id from the [catalog](https://openrouter.ai/models), as `provider/model`     |
| Max tokens   | 4096                       | Reasoning models spend part of it on thinking                                |
| Temperature  | 0.7                        |                                                                              |
| Top P        | 1.0                        |                                                                              |
| Top K        | 0 (off)                    | Not every model supports it                                                  |
| Reasoning    | Auto                       | Auto sends nothing; Off is rejected by models that always reason             |
| Timeout (s)  | 120                        |                                                                              |

## Privacy

Your messages go to OpenRouter and to the provider of the chosen model. Use the local
[Ollama](https://github.com/idia-tt/dashai-ollama) plugin if the data must stay on your machine.

## Troubleshooting

| Message                         | Cause and fix                                                          |
| ------------------------------- | ---------------------------------------------------------------------- |
| `No OpenRouter API key`         | Save the key in the OpenRouter credential                              |
| `error 401`                     | The key is invalid or disabled                                         |
| `error 402`                     | The account or the key has no credits                                  |
| `error 429`                     | Rate limited; free models have tight limits                            |
| `hit max_tokens`                | A reasoning model used the budget thinking; raise **Max tokens**       |

## Not included yet

Streaming, tools, image input and embeddings. They depend on features DashAI does not have
yet or on follow-up work in this plugin.

## Development

```bash
uv sync
uv run pytest          # unit tests, no network
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

The tests run against `stubs/`, a minimal stand-in for DashAI (see `stubs/README.md`). The
`tests/compat` smoke test needs a real DashAI and runs in the scheduled workflow.

Commits follow [Conventional Commits](https://www.conventionalcommits.org); merging to `main`
bumps the version and tags it, as in `dashai-ollama`.

## Releases

After merging to `main`, CI runs the checks, bumps the version from the conventional
commits, builds the distributions, creates a GitHub release and publishes to PyPI.
Changes limited to docs, tests or CI do not trigger a release.
If the upload fails, re-run the failed `Publish to PyPI` job in Actions; it reuses the
release artifacts without creating another version.

Before the first publication, configure the repository Actions secret `PYPI_API_TOKEN`
with a PyPI API token authorized to publish `dashai-openrouter`. The local `PYPI_AK`
credential lives in the parent workspace's `.env`; CI reads only the Actions secret.
The initial token needs account-wide scope to create the project; replace it with a
project-scoped token after the first publication. Never commit a token.

## License

MIT
