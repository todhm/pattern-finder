#!/usr/bin/env python3
"""PreToolUse hook — block Bash commands that write to secrets paths.

Receives Claude Code's PreToolUse hook JSON on stdin. If the bash
command appears to write (cp / mv / rm / tee / install / dd / shred /
truncate / shell redirect ``>``/``>>``) to a path that looks like a
secret file or lives under ``secrets/``, the hook prints a
PreToolUse deny decision JSON and exits 0. Otherwise exits 0 silently.

What counts as a "secret path":
    - any path containing ``secrets/`` segment
    - any path ending in ``.env`` (NOT ``.env.example``)
    - any path whose basename starts with ``credentials``
    - any path ending in ``.key`` / ``.pem`` / ``.token``
    - ``id_rsa`` variants

Always-allowed (intentional exceptions):
    - reads (``cat`` / ``grep`` / ``head`` / ``less`` / ``tail`` / etc.)
    - any write whose destination is a ``.example`` template

Why this exists
    On 2026-05-19 the assistant reflexively ran
    ``cp secrets/alphafolio.env.example secrets/alphafolio.env`` to
    silence a ``docker compose`` env-file error. The user had real
    secrets in that file; they got overwritten with the placeholder
    template. This hook makes that incident class impossible to repeat.
"""

from __future__ import annotations

import json
import os
import shlex
import sys


# ---- secret-path detection ----------------------------------------

SUSPICIOUS_BASENAME_PREFIXES = ("credentials", "id_rsa")
SUSPICIOUS_SUFFIXES = (".env", ".key", ".pem", ".token", ".auth")


def _is_example_template(path: str) -> bool:
    """``.example`` files are templates — writing to them is fine."""
    return path.endswith(".example") or ".example." in path


def is_secret_path(path: str) -> bool:
    if not path:
        return False
    if _is_example_template(path):
        return False

    # Strip trailing slash for consistent endswith checks.
    p = path.rstrip("/")

    # Bucket 1: anywhere under a ``secrets/`` segment.
    if "secrets/" in p or "/secrets" in p or p == "secrets":
        return True

    base = os.path.basename(p).lower()
    if not base:
        return False

    for prefix in SUSPICIOUS_BASENAME_PREFIXES:
        if base.startswith(prefix):
            return True
    for suffix in SUSPICIOUS_SUFFIXES:
        if base.endswith(suffix):
            # Special-case: don't treat ``foo.env.example`` as secret
            # (already filtered by ``_is_example_template`` above; this
            # is a safety net).
            if base.endswith(".example"):
                return False
            return True
    return False


# ---- command parsing ----------------------------------------------

WRITE_CMDS_ALL_ARGS = {"rm", "shred", "truncate"}
WRITE_CMDS_LAST_ARG = {"cp", "mv", "install", "ln"}
WRITE_CMDS_ANY_ARG = {"tee"}


def _strip_command_prefixes(tokens: list[str]) -> list[str]:
    """Drop leading ``sudo``, ``env VAR=…``, and inline VAR=val assignments."""
    out = tokens[:]
    while out:
        head = out[0]
        if head in ("sudo", "env"):
            out = out[1:]
            continue
        if "=" in head and "/" not in head and not head.startswith("-"):
            # FOO=bar style env var prefix
            out = out[1:]
            continue
        break
    return out


def _check_redirects(tokens: list[str]) -> tuple[bool, str]:
    """Detect ``> /path/to/secret`` or ``>>/path/to/secret`` redirects."""
    for i, tok in enumerate(tokens):
        if tok in (">", ">>"):
            if i + 1 < len(tokens) and is_secret_path(tokens[i + 1]):
                return True, f"redirect '{tok}' to secret path '{tokens[i + 1]}'"
        elif tok.startswith(">") and not tok.startswith(">&"):
            target = tok.lstrip(">")
            if target and is_secret_path(target):
                return True, f"redirect to secret path '{target}'"
    return False, ""


def _check_write_command(tokens: list[str]) -> tuple[bool, str]:
    """Inspect a single (possibly compound) bash sub-command."""
    if not tokens:
        return False, ""

    # Compound prefix like ``DEBUG=1 sudo cp …`` — drop prefix tokens.
    tokens = _strip_command_prefixes(tokens)
    if not tokens:
        return False, ""
    cmd0 = tokens[0]
    args = [t for t in tokens[1:] if not t.startswith("-")]

    if cmd0 in WRITE_CMDS_ALL_ARGS:
        for t in args:
            if is_secret_path(t):
                return True, f"{cmd0} targeting secret path '{t}'"
    elif cmd0 in WRITE_CMDS_LAST_ARG and args:
        dst = args[-1]
        if is_secret_path(dst):
            return True, f"{cmd0} writing to secret path '{dst}'"
    elif cmd0 in WRITE_CMDS_ANY_ARG:
        for t in args:
            if is_secret_path(t):
                return True, f"{cmd0} writing to secret path '{t}'"
    elif cmd0 == "dd":
        for t in tokens[1:]:
            if t.startswith("of=") and is_secret_path(t[len("of=") :]):
                return True, f"dd of= writing to secret path '{t[len('of='):]}'"
    return False, ""


def check_command(cmd: str) -> tuple[bool, str]:
    if not cmd:
        return False, ""

    # Split on shell control operators so each piece is inspected on its own.
    # We don't preserve semantics across pipes — that's intentional, the
    # check is conservative and per-segment.
    normalized = cmd
    for sep in [";", "&&", "||", "|"]:
        normalized = normalized.replace(sep, "\n")

    for line in normalized.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            # Unbalanced quotes — fall back to whitespace split.
            tokens = line.split()

        # Inspect this segment.
        v, why = _check_redirects(tokens)
        if v:
            return True, why
        # Filter out redirect tokens before write-command check (rm has
        # no redirect semantics but the parser still sees stray tokens).
        clean = [t for t in tokens if t not in (">", ">>") and not (t.startswith(">") and not t.startswith(">&"))]
        v, why = _check_write_command(clean)
        if v:
            return True, why
    return False, ""


# ---- hook protocol ------------------------------------------------

def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # malformed input → let through; not our role to fail closed here.

    if payload.get("tool_name") != "Bash":
        return 0

    cmd = payload.get("tool_input", {}).get("command", "")
    violation, reason = check_command(cmd)
    if not violation:
        return 0

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Blocked by .claude/scripts/block_secrets_writes.py: {reason}. "
                "Refusing to modify files under secrets/ or matching "
                "*.env / credentials* / *.key / *.pem / *.token / id_rsa*. "
                "If a secrets file needs to be (re)created, ASK THE USER to "
                "fill it in manually — never auto-copy from .example."
            ),
        }
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
