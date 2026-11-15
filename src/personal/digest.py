"""Duplicate detection and Telegram delivery for the personal job bot."""

import html
import re
import time

import requests

from src.dedup import normalize_application_url, preprocess_text

from . import profile

TELEGRAM_MAX_LENGTH = 4096
SNIPPET_LENGTH = 420
COMPACT_SNIPPET_LENGTH = 160
SEPARATOR = "\n\n━━━━━━━━━━━━━━━\n\n"
SEND_DELAY = 1.2          # seconds between messages (Telegram: ~1 msg/s per chat)
SIMILARITY_THRESHOLD = 0.8

SECTIONS = [
    ("⭐ <b>In focus</b>", lambda p: p.get("is_focus")),
    ("🔬 <b>Postdoc &amp; research staff</b>",
     lambda p: not p.get("is_focus") and p.get("is_priority_level")),
    ("🎓 <b>PhD &amp; other</b>",
     lambda p: not p.get("is_focus") and not p.get("is_priority_level")),
]


# --- duplicate detection ----------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def fingerprint(position: dict) -> dict:
    """Compact record of a vacancy used to recognise re-announcements."""
    text = preprocess_text(position.get("raw_text") or position.get("message") or "")
    return {
        "uri": position.get("uri"),
        "app_url": normalize_application_url(position.get("application_url")),
        "text": text[:400],
    }


def is_duplicate(candidate: dict, known: list[dict]) -> bool:
    """Same post, same application link, or near-identical wording."""
    cand_tokens = _tokens(candidate["text"])
    for other in known:
        if candidate["uri"] and candidate["uri"] == other.get("uri"):
            return True
        if candidate["app_url"] and candidate["app_url"] == other.get("app_url"):
            return True
        other_tokens = _tokens(other.get("text") or "")
        if len(cand_tokens) >= 6 and len(other_tokens) >= 6:
            overlap = len(cand_tokens & other_tokens) / len(cand_tokens | other_tokens)
            if overlap >= SIMILARITY_THRESHOLD:
                return True
    return False


def drop_duplicates(positions: list[dict], already_sent: list[dict]) -> list[dict]:
    """Keep the first occurrence of each vacancy not sent before.

    ``positions`` should already be ranked so the best copy wins.
    """
    known = list(already_sent)
    unique = []
    for position in positions:
        fp = fingerprint(position)
        if is_duplicate(fp, known):
            continue
        known.append(fp)
        unique.append(position)
    return unique


# --- formatting ---------------------------------------------------------------

def _tag(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value.replace("&", "")).strip("_")
    return f"#{cleaned}" if cleaned else ""


def _links(pos: dict) -> str:
    links = []
    if pos.get("url"):
        links.append(f'<a href="{html.escape(pos["url"], quote=True)}">Post</a>')
    if pos.get("application_url"):
        links.append(
            f'<a href="{html.escape(pos["application_url"], quote=True)}">Apply</a>'
        )
    return " · ".join(links)


def _snippet(pos: dict, length: int) -> str:
    text = " ".join((pos.get("raw_text") or pos.get("message") or "").split())
    if len(text) > length:
        text = text[:length - 1].rstrip() + "…"
    return text


def format_compact(pos: dict) -> str:
    """Three-line version used for vacancies outside the focus topics."""
    tags = [_tag(t) for t in pos.get("position_type") or []]
    country = pos.get("country") or ""
    if country and country != "Unknown":
        tags.append(_tag(country))
    tags += [_tag(t) for t in pos.get("wet_lab_topics") or []]
    lines = [html.escape(" ".join(t for t in tags if t))]

    title = pos.get("job_title")
    org = pos.get("hiring_organization")
    if title:
        headline = " — ".join(html.escape(x) for x in (title, org) if x)
        lines.append(f"<b>{headline}</b>")
    else:
        lines.append(html.escape(_snippet(pos, COMPACT_SNIPPET_LENGTH)))

    footer = []
    if pos.get("application_deadline"):
        footer.append(f"⏰ {html.escape(str(pos['application_deadline']))}")
    links = _links(pos)
    if links:
        footer.append(f"🔗 {links}")
    if footer:
        lines.append(" · ".join(footer))
    return "\n".join(line for line in lines if line)


def format_position(pos: dict) -> str:
    """One vacancy as a full Telegram HTML card."""
    tags = [_tag(t) for t in pos.get("position_type") or []]
    country = pos.get("country") or ""
    if country and country != "Unknown":
        tags.append(_tag(country))
    tags += [_tag(t) for t in pos.get("focus_topics") or []]
    tags += [_tag(t) for t in pos.get("wet_lab_topics") or []]
    header = " ".join(t for t in tags if t)
    if pos.get("is_focus"):
        header = f"⭐ {header}"

    lines = [html.escape(header)]

    title = pos.get("job_title")
    org = pos.get("hiring_organization")
    if title or org:
        headline = " — ".join(html.escape(x) for x in (title, org) if x)
        lines.append(f"<b>{headline}</b>")
    if pos.get("application_deadline"):
        lines.append(f"⏰ Deadline: {html.escape(str(pos['application_deadline']))}")

    text = _snippet(pos, SNIPPET_LENGTH)
    if text:
        lines.append("")
        lines.append(html.escape(text))

    links = _links(pos)
    if links:
        lines.append("")
        lines.append(f"🔗 {links}")

    return "\n".join(lines)


def summary_line(positions: list[dict]) -> str:
    """Digest title, e.g. "12 new positions · 4 in focus"."""
    total = len(positions)
    focus = sum(1 for p in positions if p.get("is_focus"))
    line = f"🧬 <b>{total} new position{'s' if total != 1 else ''}</b>"
    if focus:
        line += f" · {focus} in focus"
    return line


def build_messages(
    positions: list[dict],
    title: str | None = None,
) -> list[tuple[str, list[dict]]]:
    """Pack ranked vacancies into Telegram-sized messages, grouped by section.

    Returns (html_text, positions_in_this_message) pairs so delivery can be
    tracked per message. ``title`` is prepended to the first message.
    """
    # (text, position, joiner placed before this block when it is not first)
    blocks: list[tuple[str, dict, str]] = []
    for heading, belongs in SECTIONS:
        first = True
        for pos in positions:
            if not belongs(pos):
                continue
            compact = profile.COMPACT_NON_FOCUS and not pos.get("is_focus")
            body = format_compact(pos) if compact else format_position(pos)
            joiner = "\n\n" if compact and not first else SEPARATOR
            blocks.append((f"{heading}\n\n{body}" if first else body, pos, joiner))
            first = False
    if title and blocks:
        blocks[0] = (f"{title}\n\n{blocks[0][0]}",) + blocks[0][1:]

    messages: list[tuple[str, list[dict]]] = []
    current, current_positions = "", []
    for block, pos, joiner in blocks:
        if len(block) > TELEGRAM_MAX_LENGTH:
            block = block[:TELEGRAM_MAX_LENGTH - 1] + "…"
        candidate = block if not current else current + joiner + block
        if len(candidate) > TELEGRAM_MAX_LENGTH and current:
            messages.append((current, current_positions))
            current, current_positions = block, [pos]
        else:
            current, current_positions = candidate, current_positions + [pos]
    if current:
        messages.append((current, current_positions))
    return messages


# --- delivery -----------------------------------------------------------------

def send_message(token: str, chat_id: str, html_text: str) -> bool:
    """Send one message; waits and retries once if Telegram asks to slow down."""
    for attempt in range(2):
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": html_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
        except requests.RequestException as error:
            print(f"Telegram request failed: {error}")
            return False
        if resp.ok:
            return True
        if resp.status_code == 429 and attempt == 0:
            try:
                wait = int(resp.json()["parameters"]["retry_after"])
            except (ValueError, KeyError, TypeError):
                wait = 30
            time.sleep(min(wait, 120) + 1)
            continue
        print(f"Telegram API error: {resp.status_code} {resp.text[:300]}")
        return False
    return False


def deliver(positions: list[dict], token: str, chat_id: str) -> tuple[list[dict], list[dict]]:
    """Send the digest. Returns (delivered, not_delivered) vacancies."""
    delivered: list[dict] = []
    messages = build_messages(positions, title=summary_line(positions))
    for index, (text, in_message) in enumerate(messages):
        if not send_message(token, chat_id, text):
            remaining = [p for _, group in messages[index:] for p in group]
            return delivered, remaining
        delivered.extend(in_message)
        if index < len(messages) - 1:
            time.sleep(SEND_DELAY)
    return delivered, []
