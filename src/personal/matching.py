"""Topic and level matching for the personal job bot (pure functions, no I/O)."""

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from . import profile

_AGGREGATORS_FILE = Path(__file__).resolve().parents[2] / "docs" / "aggregators.json"


@lru_cache(maxsize=1)
def aggregator_handles() -> frozenset[str]:
    """Handles of known job-aggregator accounts (their bios list every field)."""
    try:
        data = json.loads(_AGGREGATORS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(h.lower() for h in data.get("handles", []))


@lru_cache(maxsize=None)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def split_bio(position: dict) -> tuple[str, str]:
    """Return (bio, content) for a post dict produced by BlueskySource.

    ``message`` is ``[Bio: <bio>]\\n\\n<raw_text>`` and ``metadata_text`` adds
    the quote wrapper and link preview. Content is everything except the bio.
    """
    raw_text = position.get("raw_text") or ""
    message = position.get("message") or ""
    metadata_text = position.get("metadata_text") or message

    bio = ""
    prefix = "[Bio: "
    suffix = "]\n\n" + raw_text
    if raw_text and message.startswith(prefix) and message.endswith(suffix):
        bio = message[len(prefix):len(message) - len(suffix)]
        content = raw_text + metadata_text[len(message):]
    elif message.startswith(prefix) and not raw_text:
        # No raw_text (e.g. rows loaded from elsewhere): best-effort split.
        match = re.match(r"\[Bio: (.*?)\]\n\n", message, flags=re.DOTALL)
        bio = match.group(1) if match else ""
        content = metadata_text[match.end():] if match else metadata_text
    else:
        content = raw_text + metadata_text[len(message):] if raw_text else metadata_text
    return bio, content


def _matched_tags(text: str, topics: dict[str, list[str]]) -> list[str]:
    return [
        tag for tag, patterns in topics.items()
        if any(_compile(p).search(text) for p in patterns)
    ]


def match_topics(position: dict) -> dict:
    """Find focus and wet-lab tags for a post.

    The author's bio counts too (a chromatin lab announcing "we're hiring a
    postdoc" rarely repeats its field), except for aggregator accounts.
    """
    bio, content = split_bio(position)
    handle = (position.get("user") or position.get("user_handle") or "").lower()
    text = content if handle in aggregator_handles() else f"{content}\n{bio}"
    return {
        "focus": _matched_tags(text, profile.FOCUS_TOPICS),
        "wet_lab": _matched_tags(text, profile.WET_LAB_TOPICS),
    }


def passes_prefilter(position: dict) -> bool:
    """Cheap check before spending an LLM call on a post (see profile.py)."""
    if not profile.PREFILTER_ENABLED:
        return True
    bio, content = split_bio(position)
    handle = (position.get("user") or position.get("user_handle") or "").lower()
    text = content if handle in aggregator_handles() else f"{content}\n{bio}"
    if any(_compile(p).search(text) for p in profile.LIFE_SCIENCE_PATTERNS):
        return True
    return bool(
        _matched_tags(text, profile.FOCUS_TOPICS)
        or _matched_tags(text, profile.WET_LAB_TOPICS)
    )


def _any(patterns: list[str], text: str) -> bool:
    return any(_compile(p).search(text) for p in patterns)


def triage(position: dict) -> int | None:
    """Priority of a post for the LLM: 0 is best, None means "skip it".

    0 - focus topic; 1 - research-staff wording in a life-science post;
    2 - other life-science post; 3 - anything else (only with the pre-filter
    off). Faculty-only announcements are skipped.
    """
    bio, content = split_bio(position)
    handle = (position.get("user") or position.get("user_handle") or "").lower()
    text = content if handle in aggregator_handles() else f"{content}\n{bio}"

    staff_role = _any(profile.STAFF_ROLE_PATTERNS, content)
    student_role = bool(re.search(r"\bph\.?d\b|doctoral|master", content, re.IGNORECASE))
    if _any(profile.EXCLUDED_ROLE_PATTERNS, content) and not (staff_role or student_role):
        return None

    if _matched_tags(text, profile.FOCUS_TOPICS):
        return 0
    life_science = bool(
        _matched_tags(text, profile.WET_LAB_TOPICS)
        or _any(profile.LIFE_SCIENCE_PATTERNS, text)
    )
    if not life_science:
        return None if profile.PREFILTER_ENABLED else 3
    return 1 if staff_role else 2


def deadline_passed(position: dict) -> bool:
    """True when the extracted application deadline is before today (UTC)."""
    value = str(position.get("application_deadline") or "")[:10]
    try:
        deadline = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return False
    return deadline < datetime.now(timezone.utc).date()


def is_priority_level(position: dict) -> bool:
    """True for postdoc / research-staff level positions."""
    types = position.get("position_type") or []
    return any(t in profile.PRIORITY_POSITION_TYPES for t in types)


def evaluate(position: dict) -> dict | None:
    """Decide whether a classified post belongs in the digest.

    Returns the position enriched with ``focus_topics``, ``wet_lab_topics``,
    ``is_focus``, ``is_industry`` and ``is_priority_level``, or None when it
    is not relevant.
    """
    if position.get("is_verified_job") is not True:
        return None

    is_industry = position.get("sector") == "Industry"
    if is_industry and not profile.INCLUDE_INDUSTRY:
        return None

    types = [t for t in (position.get("position_type") or [])
             if t not in profile.EXCLUDED_POSITION_TYPES]
    if not types:
        return None

    title = position.get("job_title") or ""
    if title and _any(profile.EXCLUDED_ROLE_PATTERNS, title):
        return None
    if profile.DROP_EXPIRED_DEADLINES and deadline_passed(position):
        return None

    topics = match_topics(position)
    disciplines = position.get("disciplines") or []
    relevant = (
        bool(topics["focus"])
        or bool(topics["wet_lab"])
        or any(d in profile.RELEVANT_DISCIPLINES for d in disciplines)
    )
    if not relevant:
        return None

    result = dict(position)
    result["position_type"] = types
    result["focus_topics"] = topics["focus"]
    result["wet_lab_topics"] = topics["wet_lab"]
    result["is_focus"] = bool(topics["focus"])
    result["is_industry"] = is_industry
    result["is_priority_level"] = is_priority_level(result)
    return result


def rank(positions: list[dict]) -> list[dict]:
    """Order for the digest: focus first, then research-staff level, then newest."""
    newest_first = sorted(positions, key=lambda p: p.get("created") or "", reverse=True)
    return sorted(
        newest_first,
        key=lambda p: (not p.get("is_focus"), not p.get("is_priority_level")),
    )


def all_queries(upstream_queries: list[str]) -> list[str]:
    """Upstream queries (optional) + profile queries, de-duplicated in order."""
    queries = list(upstream_queries) if profile.USE_UPSTREAM_QUERIES else []
    queries += profile.EXTRA_QUERIES
    if profile.INCLUDE_INDUSTRY:
        queries += profile.INDUSTRY_QUERIES
    seen: set[str] = set()
    unique = []
    for query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(query.strip())
    return unique
