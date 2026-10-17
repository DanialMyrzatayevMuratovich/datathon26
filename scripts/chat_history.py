#!/usr/bin/env python3
"""Condense a Claude Code session transcript (~/.claude/projects/<dir>/<session>.jsonl) into readable Markdown.

For moving work to a new chat: keeps what the user asked and what the agent answered, one line per tool call and
a mark where the user rejected a call; drops tool outputs, images and hidden reasoning, cuts <system-reminder> blocks
and masks tokens/keys. Stdlib only. Write the result to data/processed/ (gitignored): a chat is not repo content.
Usage: python3 scripts/chat_history.py <session.jsonl> <out.md> [--user-only]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

SECRET = re.compile(r"\b(ghp_|gho_|ghs_|github_pat_|hf_|sk-|xox[abp]-|AKIA)[A-Za-z0-9_\-]{10,}")
NOISE = re.compile(r"<(system-reminder|ide_opened_file)>.*?</\1>\s*", re.S)
HEADS = {"user": "### 👤 Пользователь", "assistant": "### 🤖 Агент", "summary": "### 📋 Сводка предыдущей части чата"}


def clean(text: str) -> str:
    return SECRET.sub(r"\1***", NOISE.sub("", text)).strip()


def when(ts: str | None) -> str:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone().strftime("%d.%m %H:%M")
    except ValueError:
        return ""


def tool_line(block: dict) -> str:
    inp = block.get("input") or {}
    what = next((inp[k] for k in ("description", "file_path", "command", "prompt", "pattern") if inp.get(k)), "")
    return f"→ {block.get('name')}: {' '.join(clean(str(what)).split())[:160]}"


def rejection(block: dict) -> str | None:
    body = block.get("content")
    if isinstance(body, list):
        body = " ".join(b.get("text", "") for b in body if isinstance(b, dict))
    body = str(body or "")
    if not block.get("is_error") or "doesn't want to" not in body:
        return None
    said = body.split("the user said:", 1)[1].strip() if "the user said:" in body else ""
    return "⛔ вызов отклонён пользователем" + (f": {clean(said)[:300]}" if said else "")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    user_only = "--user-only" in sys.argv
    src, out = Path(args[0]), Path(args[1])
    parts: list[str] = []
    last = None
    with src.open(encoding="utf-8") as f:
        for raw in f:
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue  # the live session may end with a half-written line
            kind = d.get("type")
            if kind not in ("user", "assistant") or d.get("isSidechain") or d.get("isMeta"):
                continue
            content = (d.get("message") or {}).get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            lines = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                kind_b = b.get("type")
                if kind_b == "text" and (txt := clean(b.get("text", ""))):
                    lines.append(txt)
                elif kind_b == "image" and kind == "user":
                    lines.append("[картинка]")
                elif kind_b == "tool_use" and not user_only:
                    lines.append(tool_line(b))
                elif kind_b == "tool_result" and (r := rejection(b)):
                    lines.append(r)
            role = "summary" if d.get("isCompactSummary") else kind
            if not lines or (user_only and role != "user"):
                continue
            typed = role == "user" and any(not ln.startswith("⛔") for ln in lines)  # each message: own heading
            if role != last or role == "summary" or typed:
                parts.append(f"\n{HEADS[role]} · {when(d.get('timestamp'))}\n")
                last = role
            parts.append("\n".join(lines))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts).lstrip() + "\n", encoding="utf-8")
    print(f"{out}: {out.stat().st_size // 1024} KB, {sum(p.startswith(chr(10) + '###') for p in parts)} turns")


if __name__ == "__main__":
    main()
