"""Small JSON state file for the personal job bot.

Committed back to the repository by the GitHub Actions workflow so that every
run continues where the previous one stopped.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

MAX_SEEN_URIS = 8000   # recent post URIs kept to avoid re-classifying
MAX_SENT = 600         # recent sent vacancies kept for duplicate detection


def utc_now_iso() -> str:
    """Timestamp in the same shape Bluesky clients use for createdAt."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class BotState:
    """Last sync position, seen URIs, sent vacancies, unsent and unclassified leftovers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.last_timestamp: str | None = None
        self.seen_uris: list[str] = []   # ordered, oldest first
        self.sent: list[dict] = []       # {uri, app_url, text, sent_at}
        self.pending: list[dict] = []    # matched vacancies not delivered yet
        self.queue: list[dict] = []      # high-priority posts not classified yet
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        self.last_timestamp = data.get("last_timestamp") or None
        self.seen_uris = [u for u in data.get("seen_uris", []) if isinstance(u, str)]
        self.sent = [s for s in data.get("sent", []) if isinstance(s, dict)]
        self.pending = [p for p in data.get("pending", []) if isinstance(p, dict)]
        self.queue = [p for p in data.get("queue", []) if isinstance(p, dict)]

    @property
    def is_first_run(self) -> bool:
        return self.last_timestamp is None

    def add_seen(self, uris) -> None:
        known = set(self.seen_uris)
        for uri in uris:
            if uri and uri not in known:
                known.add(uri)
                self.seen_uris.append(uri)
        self.seen_uris = self.seen_uris[-MAX_SEEN_URIS:]

    def advance(self, timestamp: str | None) -> None:
        """Move the sync position forward (never backwards, never into the future)."""
        if not timestamp:
            return
        timestamp = min(timestamp, utc_now_iso())
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp

    def record_sent(self, entries: list[dict]) -> None:
        self.sent = (self.sent + entries)[-MAX_SENT:]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": utc_now_iso(),
            "last_timestamp": self.last_timestamp,
            "pending": self.pending,
            "queue": self.queue,
            "sent": self.sent,
            "seen_uris": self.seen_uris,
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
