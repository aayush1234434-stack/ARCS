# Security policy

## Supported version

Security fixes are applied to the latest commit on `main`.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability involving secret
exposure, sandbox escape, or remote code execution. Email
`aayush1234434@gmail.com` with reproduction steps and the affected commit.

## Generated-code boundary

ARCS treats model-generated code as untrusted. The supported execution path is
a locked-down Docker container with networking disabled, a read-only root
filesystem, dropped Linux capabilities, resource limits, and a non-root user.

If Docker is unavailable, execution fails closed. The environment variable
`ARCS_ALLOW_UNSAFE_SUBPROCESS=1` enables a developer-only local fallback. It is
not a sandbox and must never be enabled in a shared, hosted, or public service.

## Secrets

API keys belong in `.env` for local development or in the deployment platform's
secret manager. Logs and bug reports must be reviewed for prompts, model output,
personal data, and credentials before sharing.
