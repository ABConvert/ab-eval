"""Writing `config/models.yaml` and `config/repos.yaml` from the dashboard.

A config editor that can write a file the harness cannot load is worse than no editor: it
turns a typo into a broken install and hides the moment it happened. So every path through
this module is the same four steps — refuse a credential, validate a candidate through the
harness's own loader, copy the previous version aside, then replace atomically. A save that
fails at any step leaves the file on disk byte-for-byte as it was.

Two things here are easy to get wrong and expensive to get wrong.

`config_file()` falls back to the checkout's own config when the data root does not have one,
which is right for reading and catastrophic for writing: a save through it would overwrite the
defaults the harness ships. Writes go to `config_dir()`, always — `write_path` is the only
place that decides where, and it never falls back.

The other is editing the loaded pydantic objects and dumping those back. `load_repos` expands
`ABEVAL_PATH_*` over `local_path`, so a round trip through `RepoConfig` would write this
machine's absolute path over the portable one the team committed. Every edit here mutates the
plain dict `yaml.safe_load` returns, and pydantic only ever sees a temporary file.
"""

from __future__ import annotations

import ipaddress
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval_harness.paths import config_dir, config_file

# The two files this module will write. Anything else is rejected before a path is built.
FILES = ("models.yaml", "repos.yaml")

# Providers with an adapter in `adapters/registry.py`. Offered as a closed list because a
# provider with no adapter is a config that parses and then fails at the start of a run.
PROVIDERS = ("anthropic", "claude-code", "openai-compat", "codex-cli")

# Which of the fields this form gates each provider's adapter can actually act on. A box is
# offered only where the value reaches the request, because a field that saves and is never
# sent is the defect this page exists to prevent — the same one that had two bench-v2 runs
# configured `effort: high` against endpoints that were never told.
#
#   max_output_tokens  reaches claude-code through CLAUDE_CODE_MAX_OUTPUT_TOKENS on the CLI
#                      the SDK spawns; `codex exec` has no equivalent at all.
#   temperature        only the openai-compat body carries one: the Anthropic Messages API
#                      has no such parameter and codex exposes no override.
#   reasoning_param    says how an OpenAI-shaped endpoint wants to be told how hard to think.
#                      The other three are told through their own SDK, from `effort`.
#
# base_url and api_key_env are not in here on purpose: they are labelled rather than gated,
# because an entry can legitimately carry one for a provider that ignores it and a form that
# hid them would be a form that deleted them.
PROVIDER_FIELDS: dict[str, tuple[str, ...]] = {
    "anthropic": ("max_output_tokens",),
    "claude-code": ("max_output_tokens",),
    "openai-compat": ("max_output_tokens", "temperature", "reasoning_param"),
    "codex-cli": (),
}

# Every field PROVIDER_FIELDS gates. A provider the table does not know — hand-edited, no
# adapter yet — gets them all rather than having its entry quietly stripped on first save.
GATED_FIELDS = tuple(dict.fromkeys(f for fs in PROVIDER_FIELDS.values() for f in fs))


def fields_for(provider: str) -> tuple[str, ...]:
    """The gated fields this provider can use; all of them for one the table does not know."""
    return PROVIDER_FIELDS.get(provider, GATED_FIELDS)


# models.yaml keys the harness reads for itself rather than putting on trial.
SPECIAL_KEYS = {
    "judge": (
        "Scores code quality on the cases that resolve, and is what `eval-harness review` "
        "reads — that command fails without it. Scoring without one skips the quality metric."
    ),
    "classifier": (
        "Labels a pull request as bug fix or feature while collecting, when the pull request "
        "carries no label of its own. Without it those pull requests are skipped."
    ),
}

# An environment variable name, which is all `api_key_env` may ever hold.
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Credential shapes, matched against the whole candidate text. Deliberately anchored on issuer
# prefixes rather than on entropy: a false positive here refuses a legitimate save, and
# `base_url`, image tags and commit shas are all long opaque strings that must keep working.
KEY_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bxai-[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{30,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
)


class ConfigError(Exception):
    """A candidate that must not be written. The message is shown in the form."""


def is_loopback(host: str) -> bool:
    """Whether a bind address only this machine can reach.

    `Runner.full` is a shell command the harness later executes and `dockerfile` names what
    gets built, so writing repos.yaml from a browser is equivalent to writing code that will
    run here. On loopback that is exactly as privileged as editing the file in an editor.
    Bound anywhere else it is remote code execution, so the write routes refuse.

    A hostname other than `localhost` is not resolved: a name that happens to point at 127.0.0.1
    today is not a guarantee, and guessing wrong grants the thing this check exists to withhold.
    """
    h = host.strip().strip("[]")
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def write_path(name: str) -> Path:
    """Where a save lands. The data root's own config, never the checkout's fallback."""
    if name not in FILES:
        raise ConfigError(f"{name} is not a file this page writes")
    return config_dir() / name


def read_path(name: str) -> Path:
    """Where the page reads from, which on a fresh data root is the checkout's own copy."""
    if name not in FILES:
        raise ConfigError(f"{name} is not a file this page reads")
    return config_file(name)


def load_raw(name: str) -> dict[str, Any]:
    """The file as plain data, or `{}` when it does not exist yet.

    Deliberately not `load_models` / `load_repos`: this is what an edit is applied to, and it
    has to survive a file that does not validate — you cannot fix an invalid entry through a
    form that refuses to load it.
    """
    path = read_path(name)
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} does not parse as YAML: {e}") from e
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping of name to settings, not {type(raw).__name__}")
    return raw


class _Dumper(yaml.SafeDumper):
    """SafeDumper that indents list items under their key, the way people write YAML.

    PyYAML's default puts `- item` in the same column as the key above it, which is valid
    and reads as though the list belongs to the parent. These files are edited by hand as
    often as by this page, so what it writes has to look like what a person would have.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def dump(data: dict[str, Any]) -> str:
    """Serialise an edited mapping back to YAML, keeping the order the entries were in.

    Lossy by nature — comments and inline `{a: b}` style do not survive a parse, so a form
    save normalises the whole file once. That is the cost of a form, it is why the backup
    exists, it is why the YAML editor is there for anyone who wants their formatting kept,
    and the page says all three before you click.
    """
    return yaml.dump(
        data, Dumper=_Dumper, sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def tidy_number(value: float) -> float | int:
    """4.0 -> 4, 2.5 -> 2.5. A form posts strings and floats come back from every one of them.

    Without this, saving one unrelated field rewrites `cpus: 4` as `cpus: 4.0` — a diff in a
    version-controlled file that says nothing happened, which is the kind of noise that
    teaches people to stop reading diffs.
    """
    return int(value) if float(value).is_integer() else value


def find_secret(text: str) -> str | None:
    """The issuer prefix of the first credential-shaped run in `text`, if there is one.

    Returns the prefix, never the match: a message that quoted the key would put it in the
    page, the logs and the browser history, which is the thing being prevented.
    """
    for shape in KEY_SHAPES:
        m = shape.search(text)
        if m:
            return m.group(0)[:8]
    return None


def reject_secrets(text: str) -> None:
    prefix = find_secret(text)
    if prefix is not None:
        raise ConfigError(
            f"This looks like an API key (it starts {prefix}…), and these files are committed "
            "to version control. They record the NAME of the environment variable holding a "
            "key, never the key. Export the key in the shell that starts the harness and put "
            "the variable's name here instead. Nothing was written."
        )


def redact(body: dict[str, Any]) -> dict[str, Any]:
    """The same fields with anything credential-shaped removed.

    A refused save hands back what the reader typed so they do not retype eight fields to fix
    one — but if what they typed was a key, handing it back puts the key in the page, which is
    the thing being refused. The one field they must retype is the one they should not have
    pasted.
    """
    out: dict[str, Any] = {}
    for key, value in body.items():
        if isinstance(value, str) and find_secret(value) is not None:
            continue
        if isinstance(value, dict):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


def duplicate_keys(text: str) -> list[str]:
    """Top-level names defined more than once.

    `yaml.safe_load` keeps the last of a repeated key and says nothing, so a file that
    accidentally defines `judge` twice silently loses one of them. Caught here because the
    YAML editor is exactly where that happens — pasting a drafted entry below an existing one.
    """
    try:
        node = yaml.compose(text)
    except yaml.YAMLError:
        return []  # it does not parse at all; the loader will say so, with a line number
    if not isinstance(node, yaml.MappingNode):
        return []
    seen: set[str] = set()
    dupes: list[str] = []
    for key_node, _ in node.value:
        key = getattr(key_node, "value", None)
        if not isinstance(key, str):
            continue
        if key in seen and key not in dupes:
            dupes.append(key)
        seen.add(key)
    return dupes


def _reject_bad_env_names(models: dict[str, Any]) -> None:
    """`api_key_env` holds a variable name, and pydantic types it as a plain string.

    So the model validating is not enough — raw YAML with a key pasted into that field
    validates perfectly. This is the check that closes it, and it runs on both the form path
    and the YAML path because both can reach the field.
    """
    for key, cfg in models.items():
        var = getattr(cfg, "api_key_env", None)
        if var is None or ENV_NAME.match(var):
            continue
        if find_secret(var) is not None:
            raise ConfigError(
                f"{key}: that is an API key, and api_key_env holds the NAME of the environment "
                "variable the key is in. Expected something like ANTHROPIC_API_KEY. Nothing "
                "was written."
            )
        # Not a shape anyone recognises, which is its own reason to refuse: an unrecognised
        # credential looks exactly like this, and a name that is not a name cannot be read
        # from the environment anyway.
        raise ConfigError(
            f"{key}: api_key_env must be an environment variable name — upper case letters, "
            f"digits and underscores, like ANTHROPIC_API_KEY. {var!r} is not one, and if it "
            "is the key itself, export it in your shell and name the variable here instead. "
            "Nothing was written."
        )


def stage(name: str, text: str) -> tuple[Path, dict[str, Any]]:
    """Write a candidate to a temp file beside its destination and load it from there.

    Returns the staged file and what the loader made of it. What is validated is then the
    exact bytes that land, because `save` renames this same file into place rather than
    writing the text a second time.

    The caller owns the returned path: rename it or unlink it.
    """
    from eval_harness.config import load_models, load_repos

    reject_secrets(text)
    dupes = duplicate_keys(text)
    if dupes:
        raise ConfigError(
            "Defined twice: "
            + ", ".join(dupes)
            + ". YAML keeps only the last of a repeated name, so one of them would be lost "
            "without a word. Rename or delete one. Nothing was written."
        )

    target = write_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        loaded: dict[str, Any]
        if name == "models.yaml":
            loaded = dict(load_models(tmp))
            _reject_bad_env_names(loaded)
        else:
            loaded = dict(load_repos(tmp))
        if not loaded:
            raise ConfigError(
                f"{name} defines nothing. An empty file is not a configuration — the harness "
                "would report it as missing. Nothing was written."
            )
    except ConfigError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise ConfigError(str(e)) from e
    return tmp, loaded


def validate(name: str, text: str) -> dict[str, Any]:
    """What the loader makes of a candidate, staging and discarding a file to find out."""
    tmp, loaded = stage(name, text)
    tmp.unlink(missing_ok=True)
    return loaded


def save(name: str, text: str) -> Path | None:
    """Validate, back up, and replace. Returns the backup written, or None on a first write.

    The backup is not housekeeping. These files are how someone's install works, and a stray
    command has emptied one before; the cost of keeping the last version beside it is a few
    kilobytes, and the cost of not keeping it is an afternoon.
    """
    tmp, _ = stage(name, text)  # raises before anything on disk has been touched
    target = write_path(name)
    try:
        backup: Path | None = None
        if target.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            # Not `.yaml`: a backup that still looks like config is a backup someone will load.
            backup = target.with_name(f"{target.name}.{stamp}.bak")
            backup.write_bytes(target.read_bytes())
    except OSError:
        tmp.unlink(missing_ok=True)  # no backup, no write: the file keeps the version it has
        raise
    # Atomic within the directory: a reader sees one whole version or the other, never a
    # half-written file — and it is the file the loader just accepted, byte for byte.
    os.replace(tmp, target)
    return backup


def backups(name: str) -> list[Path]:
    """Every backup of one file, newest first. The timestamp sorts lexically."""
    target = write_path(name)
    return sorted(target.parent.glob(f"{target.name}.*.bak"), reverse=True)


# ---------------------------------------------------------------- edits, as dict surgery


def put_model(
    key: str, body: dict[str, Any], *, replacing: str = "", keep: tuple[str, ...] = ()
) -> Path | None:
    """Add or update one entry in models.yaml, keeping the order of the others.

    `replacing` is the key the form was opened on: renaming an entry has to remove the old name
    in the same write, or a rename silently becomes a duplicate.

    `keep` names fields the form did not offer for the chosen provider, carried over from the
    entry being replaced. Not offering a field is the point; deleting the line someone typed
    because a later save could not show it back to them is not.
    """
    data = load_raw("models.yaml")
    previous = data.get(replacing or key)
    if isinstance(previous, dict):
        for field in keep:
            if field not in body and field in previous:
                body[field] = previous[field]
    if replacing and replacing != key:
        data.pop(replacing, None)
    data[key] = body
    return save("models.yaml", dump(data))


def drop_model(key: str) -> Path | None:
    data = load_raw("models.yaml")
    if key not in data:
        raise ConfigError(f"{key} is not in models.yaml; nothing to remove.")
    if len(data) == 1:
        raise ConfigError(
            f"{key} is the only entry. Removing it would leave a file the harness reports as "
            "missing — delete the file yourself if that is what you want."
        )
    del data[key]
    return save("models.yaml", dump(data))


def patch_repo(key: str, fields: dict[str, Any]) -> Path | None:
    """Update the shallow fields of one repository, leaving everything nested untouched.

    A merge rather than a replace, because the form covers a fraction of the entry: dep_dirs,
    runners and the glob lists are not on it and must survive a save from it.
    """
    data = load_raw("repos.yaml")
    entry = data.get(key)
    if not isinstance(entry, dict):
        raise ConfigError(
            f"{key} is not a repository in repos.yaml. Add one in the YAML editor — a new "
            "repository needs dependency roots and runners, which the form does not cover."
        )
    entry.update(fields)
    return save("repos.yaml", dump(data))
