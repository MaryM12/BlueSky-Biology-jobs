"""Personal job bot entry point.

    python -m src.personal.run              # fetch, classify, send to Telegram
    python -m src.personal.run --dry-run    # print the digest, change nothing

Required environment: BLUESKY_HANDLE, BLUESKY_PASSWORD, one LLM key
(MISTRAL_API_KEY recommended), TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID.
Tune the search in src/personal/profile.py. Setup guide: MYBOT.md.
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.llm import LLMUnavailableError
from src.logger import setup_logger

from . import profile
from .digest import build_messages, deliver, drop_duplicates, fingerprint, summary_line
from .llm import PersonalLLM
from .matching import all_queries, evaluate, passes_prefilter, rank
from .state import BotState, utc_now_iso

logger = setup_logger()

DEFAULT_STATE_FILE = "mybot_data/state.json"
DEFAULT_ARCHIVE_FILE = "mybot_data/positions.csv"

# Fields kept for delivery / archive (everything else is dropped after matching).
KEPT_FIELDS = [
    "uri", "url", "user", "created", "raw_text",
    "position_type", "country", "disciplines", "sector", "is_industry",
    "job_title", "hiring_organization", "application_url",
    "application_deadline", "location_text",
    "focus_topics", "wet_lab_topics", "is_focus", "is_priority_level",
]
ARCHIVE_FIELDS = [
    "sent_at", "created", "is_focus", "topics", "sector", "position_type", "country",
    "job_title", "hiring_organization", "application_deadline",
    "application_url", "url", "user", "text",
]


def slim(position: dict) -> dict:
    """Keep only what delivery and the archive need."""
    kept = {key: position.get(key) for key in KEPT_FIELDS}
    kept["raw_text"] = position.get("raw_text") or position.get("message") or ""
    return kept


def first_run_since(days: int) -> str:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    return start.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def classify_posts(posts, classifier, delay: float, max_classify: int,
                   max_minutes: float = 0):
    """Classify posts oldest-first.

    Posts that fail the keyword pre-filter are skipped without an LLM call.
    Returns (matched, processed_posts, complete, llm_calls). ``complete`` is
    False when the LLM became unavailable, the per-run cap of LLM calls was
    reached or the time budget ran out (some free LLM tiers allow only a few
    requests per minute); the remaining posts are picked up by the next run.
    """
    ordered = sorted(posts, key=lambda p: p.created_at)
    matched, processed = [], []
    complete = True
    llm_calls = 0
    started = time.monotonic()
    for index, post in enumerate(ordered):
        data = post.to_dict()
        if not passes_prefilter(data):
            processed.append(post)
            continue
        if max_minutes > 0 and time.monotonic() - started > max_minutes * 60:
            logger.warning(
                f"Time budget of {max_minutes:g} min used up; "
                f"{len(ordered) - index} posts left for the next run"
            )
            complete = False
            break
        if llm_calls >= max_classify:
            logger.warning(
                f"Per-run cap of {max_classify} LLM calls reached; "
                f"{len(ordered) - index} posts left for the next run"
            )
            complete = False
            break
        if delay > 0 and llm_calls > 0:
            time.sleep(delay)
        llm_calls += 1
        try:
            result = classifier.classify_post(
                data.get("raw_text") or data.get("message", ""),
                metadata_text=data.get("metadata_text"),
                posted_at=data.get("created"),
            )
        except LLMUnavailableError as error:
            logger.error(f"LLM unavailable, stopping early: {error}")
            complete = False
            break
        data.update(result)
        data["sector"] = getattr(classifier.llm, "last_sector", "Academia")
        processed.append(post)
        evaluated = evaluate(data)
        if evaluated:
            matched.append(slim(evaluated))
    return matched, processed, complete, llm_calls


def add_groq_fallbacks(classifier) -> None:
    """Chain extra Groq models behind the configured provider(s)."""
    from src.llm.fallback import FallbackProvider
    from src.llm.groq import GroqProvider

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return
    llm = classifier.llm
    for model in profile.GROQ_FALLBACK_MODELS:
        extra = GroqProvider(key, model=model)
        extra.name = f"Groq {model}"
        llm = FallbackProvider(llm, extra)
    classifier.llm = llm


def default_llm_delay() -> float:
    """Seconds between LLM calls: slower when Groq's 8k tokens/min is the limit."""
    other_keys = ("MISTRAL_API_KEY", "GEMINI_API_KEY", "NVIDIA_API_KEY")
    groq_only = os.environ.get("GROQ_API_KEY") and not any(
        os.environ.get(k) for k in other_keys
    )
    return 7.0 if groq_only else 1.0


def append_archive(path: Path, positions: list[dict], sent_at: str) -> None:
    """Append delivered vacancies to a CSV you can browse on GitHub."""
    if not positions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARCHIVE_FIELDS)
        if is_new:
            writer.writeheader()
        for pos in positions:
            topics = (pos.get("focus_topics") or []) + (pos.get("wet_lab_topics") or [])
            writer.writerow({
                "sent_at": sent_at,
                "created": pos.get("created"),
                "is_focus": "yes" if pos.get("is_focus") else "",
                "topics": ", ".join(topics),
                "sector": pos.get("sector") or "Academia",
                "position_type": ", ".join(pos.get("position_type") or []),
                "country": pos.get("country"),
                "job_title": pos.get("job_title"),
                "hiring_organization": pos.get("hiring_organization"),
                "application_deadline": pos.get("application_deadline"),
                "application_url": pos.get("application_url"),
                "url": pos.get("url"),
                "user": pos.get("user"),
                "text": " ".join((pos.get("raw_text") or "").split()),
            })


def _plain(html_text: str) -> str:
    return re.sub(r"<[^>]+>", "", html_text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Personal Bluesky job bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the digest; do not send or save state")
    parser.add_argument("-l", "--limit", type=int, default=50,
                        help="Results per Bluesky query, 1-100 (default: 50)")
    parser.add_argument("--lookback-days", type=int,
                        default=profile.FIRST_RUN_LOOKBACK_DAYS,
                        help="First run only: how many days back to look")
    parser.add_argument("--max-classify", type=int, default=1200,
                        help="Max LLM calls per run (default: 1200)")
    parser.add_argument("--max-minutes", type=float, default=140,
                        help="Stop classifying after this many minutes and "
                             "continue in the next run (default: 140, 0 = no limit)")
    parser.add_argument("--state", default=DEFAULT_STATE_FILE)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_FILE)
    args = parser.parse_args(argv)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHANNEL_ID", "")
    if not args.dry_run and not (token and chat_id):
        logger.error("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID (or use --dry-run)")
        return 2

    # Imported lazily: both modules log in / read keys from the environment.
    from bluesky_search import get_classifier
    from src.sources.bluesky import BlueskySource, DEFAULT_QUERIES

    classifier = get_classifier()
    if classifier is None:
        logger.error("No LLM key configured (set MISTRAL_API_KEY); the bot needs it "
                     "to tell real vacancies from chatter")
        return 2
    add_groq_fallbacks(classifier)
    # Adds the profile's extra rules (research staff, industry, sector) to the
    # upstream prompt without modifying the upstream classifier.
    classifier.llm = PersonalLLM(classifier.llm)

    state = BotState(args.state)
    if state.is_first_run:
        since = first_run_since(args.lookback_days)
        logger.info(f"First run: looking back {args.lookback_days} days (since {since})")
    else:
        since = state.last_timestamp
        logger.info(f"Incremental run since {since}")

    queries = all_queries(DEFAULT_QUERIES)
    logger.info(f"Searching Bluesky with {len(queries)} queries")
    source = BlueskySource(queries=queries, limit=max(1, min(args.limit, 100)))
    posts, seen_uris = source.fetch_posts(
        since_timestamp=since,
        existing_uris=set(state.seen_uris),
    )
    logger.info(f"{len(posts)} new posts ({classifier.llm.name})")

    delay = float(os.environ.get("MYBOT_LLM_DELAY") or default_llm_delay())
    matched, processed, complete, llm_calls = classify_posts(
        posts, classifier, delay, args.max_classify, args.max_minutes
    )
    logger.info(
        f"{len(processed)} posts handled: {len(processed) - llm_calls} skipped by the "
        f"pre-filter, {llm_calls} sent to the LLM, {len(matched)} match the profile"
    )

    if complete:
        state.add_seen(seen_uris)
    else:
        state.add_seen(p.uri for p in processed)
        state.add_seen(p.quoted_uri for p in processed if p.quoted_uri)
    if processed:
        state.advance(max(p.created_at for p in processed))

    candidates = rank(state.pending + matched)
    to_send = drop_duplicates(candidates, state.sent)
    logger.info(f"{len(to_send)} vacancies to send "
                f"({len(candidates) - len(to_send)} duplicates dropped)")

    if args.dry_run:
        for text, _ in build_messages(to_send, title=summary_line(to_send)):
            print("\n" + _plain(text))
        logger.info("Dry run: nothing sent, state not saved")
        return 0

    delivered, remaining = deliver(to_send, token, chat_id) if to_send else ([], [])
    sent_at = utc_now_iso()
    state.record_sent([{**fingerprint(p), "sent_at": sent_at} for p in delivered])
    state.pending = remaining
    append_archive(Path(args.archive), delivered, sent_at)
    state.save()

    if remaining:
        logger.error(f"Telegram delivery failed; {len(remaining)} vacancies kept "
                     "for the next run")
        return 1
    logger.info(f"Sent {len(delivered)} vacancies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
