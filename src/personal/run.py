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
from .matching import all_queries, evaluate, rank
from .state import BotState, utc_now_iso

logger = setup_logger()

DEFAULT_STATE_FILE = "mybot_data/state.json"
DEFAULT_ARCHIVE_FILE = "mybot_data/positions.csv"

# Fields kept for delivery / archive (everything else is dropped after matching).
KEPT_FIELDS = [
    "uri", "url", "user", "created", "raw_text",
    "position_type", "country", "disciplines",
    "job_title", "hiring_organization", "application_url",
    "application_deadline", "location_text",
    "focus_topics", "wet_lab_topics", "is_focus", "is_priority_level",
]
ARCHIVE_FIELDS = [
    "sent_at", "created", "is_focus", "topics", "position_type", "country",
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


def classify_posts(posts, classifier, delay: float, max_classify: int):
    """Classify posts oldest-first.

    Returns (matched, processed_posts, complete). ``complete`` is False when
    the LLM became unavailable or the per-run cap was reached; the remaining
    posts are picked up by the next run.
    """
    ordered = sorted(posts, key=lambda p: p.created_at)
    matched, processed = [], []
    complete = True
    for index, post in enumerate(ordered):
        if index >= max_classify:
            logger.warning(
                f"Per-run cap of {max_classify} posts reached; "
                f"{len(ordered) - index} left for the next run"
            )
            complete = False
            break
        data = post.to_dict()
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
        processed.append(post)
        evaluated = evaluate(data)
        if evaluated:
            matched.append(slim(evaluated))
        if delay > 0 and index < len(ordered) - 1:
            time.sleep(delay)
    return matched, processed, complete


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
                        help="Max posts sent to the LLM per run (default: 1200)")
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
    logger.info(f"{len(posts)} new posts to classify ({classifier.llm.name})")

    delay = float(os.environ.get("MYBOT_LLM_DELAY", "1.0"))
    matched, processed, complete = classify_posts(
        posts, classifier, delay, args.max_classify
    )
    logger.info(f"{len(matched)} of {len(processed)} classified posts match the profile")

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
