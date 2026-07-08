# Security Policy

## Scope

QuantThink is a local benchmarking tool. It does not expose network services or handle user authentication. Security concerns are limited to:

- Local file handling (result files, config YAML, frozen eval subsets)
- Optional remote model endpoints (OpenAI-compatible backend)
- Sandboxed execution of model-generated code (E5 LiveCodeBench tier, stretch only) — runs in an isolated, network-disabled, resource-limited sandbox and never against the repo tree or home directory

## Reporting a Vulnerability

To report a security issue, open a GitHub issue with the label `security`. For sensitive reports, use the GitHub private security advisory feature.

## Dependencies

Keep dependencies up to date. Run `uv sync` to install pinned versions. Review `uv.lock` before deploying in shared environments.

## Model Endpoints

When using the `openai` backend with a remote endpoint, keep API keys in environment variables — never in config YAML files committed to version control.
