# Security

This tool runs code written by a language model. That is its purpose, and it is the first thing
to understand before running it.

## The sandbox

Every tool call a model makes executes with `docker exec` inside a container created for that
one attempt:

| Property | How |
|---|---|
| No network | `--network none` for the whole attempt |
| No credentials | nothing from your environment is passed in; no API keys, no tokens, no SSH agent |
| No host filesystem | source arrives as a tar stream via `git archive` and `docker cp`, never a bind mount |
| Bounded | CPU, memory and pid limits from `limits` in `repos.yaml` |
| Reduced privileges | capabilities dropped at creation |
| Disposable | the container is removed when the attempt ends, whatever the outcome |

Your working copy of the target repository is only ever read, with `git archive` against a
commit. The harness never checks out, never writes, and never runs a command in it.

Dependencies are the one exception: a separate prep container runs your install command
**with** network access, because that is what `npm ci` and `uv sync` need. It runs before the
model does, holds no model output, and writes only into named volumes. Only this prep step
writes the shared dependency volumes. Each attempt receives disposable,
writable copies; the copy step mounts the originals read-only with no network. Removing an
attempt also removes its copies, so model changes cannot poison later attempts. Copying large
dependency trees adds startup time and temporary disk use. The hardened cache uses a new
`abeval-deps-v2-` prefix so it never reuses volumes writable by an older harness.

Trust the repository and its install scripts: the network-enabled prep step executes them.

## What is still your risk

- **Model output leaves the sandbox as a diff.** Read it before applying anything.
- **The prep step executes your repository's install scripts.** Point this at repositories you
  trust, on the same basis you would `npm install` them.
- **Ticket and PR text is sent to a model provider.** `collect` scans the task before
  classification or writing a case, including on dry runs.
  The classifier also sanitizes its own input. Matching API keys (including project and service
  account keys), tokens, private keys and JWTs stop collection of that case; this is not a
  guarantee that all secrets are detected. Raw tracker caches and source files remain private.
  Cases from a private repository contain your source code — treat a case file with
  the same care as the repository it came from.
- **Provider credentials are yours.** They live in your environment or your provider's CLI login.
  The harness reads the names of environment variables from config and never writes a value
  anywhere, including into logs and records.

## The dashboard

The dashboard contains private tickets, source diffs and configuration. It accepts only loopback
bind addresses; remote access requires an SSH tunnel. Local Host validation prevents a foreign
hostname from reaching the app through DNS rebinding. Every mutating route requires an exact
same-origin Origin header and rejects cross-site Fetch Metadata before touching data or jobs.
Missing Origin is rejected too; command-line clients must set it explicitly. A link from another
site can still open the dashboard normally.

This protects against remote sites, not another process running as your local user.

## Reporting a vulnerability

Open a private security advisory through GitHub's "Report a vulnerability" on this repository.
Please do not open a public issue for anything that looks like a sandbox escape.

Include what you ran, the `repos.yaml` involved, and what escaped. A reproduction against a
public repository helps most.

We will acknowledge within a week and aim to have a fix or a documented mitigation within 30
days.
