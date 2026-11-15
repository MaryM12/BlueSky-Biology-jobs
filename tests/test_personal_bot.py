"""Tests for the personal job bot (src/personal). No network."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.llm import JobClassifier, LLMUnavailableError
from src.llm.base import LLMProvider
from src.personal import digest, matching, profile, run
from src.personal.llm import PersonalLLM, parse_sector
from src.personal.state import BotState, MAX_SEEN_URIS, utc_now_iso
from src.sources.base import Post


def make_position(text, bio="", user="lab.bsky.social", **fields):
    message = f"[Bio: {bio}]\n\n{text}" if bio else text
    position = {
        "uri": fields.pop("uri", f"at://did/{abs(hash(text))}"),
        "message": message,
        "raw_text": text,
        "metadata_text": message + fields.pop("extra", ""),
        "url": "https://bsky.app/profile/lab/post/1",
        "user": user,
        "created": "2026-11-10T10:00:00.000Z",
        "is_verified_job": True,
        "disciplines": ["Biology"],
        "country": "Germany",
        "position_type": ["Postdoc"],
    }
    position.update(fields)
    return position


# --- topic matching ---------------------------------------------------------

@pytest.mark.parametrize("text,tag", [
    ("Postdoc on Polycomb-mediated repression", "Epigenetics"),
    ("We study chromatin and histone marks", "Epigenetics"),
    ("PRC2 biochemistry postdoc", "Epigenetics"),
    ("Hi-C and Micro-C of mouse cortex", "Genome3D"),
    ("postdoc in 3D genome organisation", "Genome3D"),
    ("how TADs and CTCF shape expression", "Genome3D"),
    ("epigenetics of human neurons", "Neuro"),
    ("iPSC-derived brain organoids", "Neuro"),
    ("microglia in neurodegeneration", "Neuro"),
])
def test_focus_patterns_match(text, tag):
    assert tag in matching.match_topics(make_position(text))["focus"]


@pytest.mark.parametrize("text", [
    "PhD position in deep neural networks for vision",
    "Postdoc in medieval history, a tad unusual",
    "Research assistant in labour economics",
    "this chic new lab is hiring",
])
def test_focus_patterns_do_not_overmatch(text):
    assert matching.match_topics(make_position(text))["focus"] == []


def test_wet_lab_tags():
    topics = matching.match_topics(
        make_position("Postdoc in molecular biology and cell biology, CRISPR screens")
    )
    assert topics["wet_lab"] == ["MolBio", "CellBio"]


def test_bio_counts_for_labs_but_not_aggregators():
    lab = make_position("We are hiring a postdoc! Apply via the link.",
                        bio="Chromatin biology lab at EMBL")
    assert matching.match_topics(lab)["focus"] == ["Epigenetics"]

    aggregator = sorted(matching.aggregator_handles())[0]
    agg = make_position("We are hiring a postdoc! Apply via the link.",
                        bio="Jobs in chromatin, neurons, everything", user=aggregator)
    assert matching.match_topics(agg)["focus"] == []


def test_link_preview_counts():
    pos = make_position("New opening in the lab!",
                        extra="\n\n[Linked page - Postdoc in epigenomics: apply now]")
    assert matching.match_topics(pos)["focus"] == ["Epigenetics"]


# --- relevance and ranking --------------------------------------------------

def test_evaluate_rejects_non_jobs_and_off_topic():
    assert matching.evaluate(make_position("chromatin", is_verified_job=False)) is None
    off_topic = make_position("Postdoc in labour economics", disciplines=["Economics"])
    assert matching.evaluate(off_topic) is None


def test_evaluate_keeps_plain_biology_without_star():
    result = matching.evaluate(make_position("Postdoc in plant ecology genetics"))
    assert result is not None
    assert result["is_focus"] is False
    assert result["is_priority_level"] is True


def test_evaluate_keeps_focus_topic_outside_biology():
    pos = make_position("Postdoc: DNA methylation in cancer cohorts", disciplines=["Medicine"])
    result = matching.evaluate(pos)
    assert result["is_focus"] is True
    assert result["focus_topics"] == ["Epigenetics"]


def test_phd_positions_stay_but_are_not_priority():
    result = matching.evaluate(make_position("PhD position in chromatin biology",
                                             position_type=["PhD Student"]))
    assert result is not None
    assert result["is_priority_level"] is False


def test_excluded_position_types(monkeypatch):
    monkeypatch.setattr(profile, "EXCLUDED_POSITION_TYPES", {"Master Student"})
    assert matching.evaluate(make_position("MSc thesis on chromatin",
                                           position_type=["Master Student"])) is None
    mixed = matching.evaluate(make_position("MSc and postdoc on chromatin",
                                            position_type=["Master Student", "Postdoc"]))
    assert mixed["position_type"] == ["Postdoc"]


def test_rank_focus_then_level_then_newest():
    def pos(name, focus, priority, created):
        return {"name": name, "is_focus": focus, "is_priority_level": priority,
                "created": created}
    ranked = matching.rank([
        pos("phd-plain", False, False, "2026-11-12"),
        pos("postdoc-plain", False, True, "2026-11-12"),
        pos("phd-focus", True, False, "2026-11-12"),
        pos("postdoc-focus-old", True, True, "2026-11-01"),
        pos("postdoc-focus-new", True, True, "2026-11-12"),
    ])
    assert [p["name"] for p in ranked] == [
        "postdoc-focus-new", "postdoc-focus-old", "phd-focus",
        "postdoc-plain", "phd-plain",
    ]


def test_all_queries_merges_and_dedupes(monkeypatch):
    monkeypatch.setattr(profile, "EXTRA_QUERIES", ["postdoc chromatin", "PhD Position "])
    monkeypatch.setattr(profile, "INDUSTRY_QUERIES", ["scientist chromatin"])
    assert matching.all_queries(["PhD position", "join my lab"]) == [
        "PhD position", "join my lab", "postdoc chromatin", "scientist chromatin",
    ]
    monkeypatch.setattr(profile, "INCLUDE_INDUSTRY", False)
    monkeypatch.setattr(profile, "USE_UPSTREAM_QUERIES", False)
    assert matching.all_queries(["PhD position"]) == ["postdoc chromatin", "PhD Position"]


def test_profile_patterns_compile():
    for topics in (profile.FOCUS_TOPICS, profile.WET_LAB_TOPICS):
        for patterns in topics.values():
            for pattern in patterns:
                matching._compile(pattern)


# --- duplicates -------------------------------------------------------------

def test_duplicates_by_application_url_and_text():
    first = make_position("Postdoc in chromatin biology at EMBL Heidelberg, apply by Dec 1",
                          uri="at://a", application_url="https://embl.org/jobs/1?utm_source=x")
    same_link = make_position("Reminder: one week left for our position!",
                              uri="at://b", application_url="http://embl.org/jobs/1/")
    same_text = make_position("Postdoc in chromatin biology at EMBL Heidelberg, apply by Dec 1 !!",
                              uri="at://c")
    other = make_position("PhD position on yeast metabolism at ETH Zurich, start in spring",
                          uri="at://d")
    kept = digest.drop_duplicates([first, same_link, same_text, other], already_sent=[])
    assert [p["uri"] for p in kept] == ["at://a", "at://d"]


def test_duplicates_against_previously_sent():
    pos = make_position("Postdoc in chromatin biology at EMBL Heidelberg", uri="at://a")
    sent = [digest.fingerprint(pos)]
    assert digest.drop_duplicates([pos], sent) == []


# --- formatting -------------------------------------------------------------

def test_format_position_escapes_and_tags():
    pos = matching.evaluate(make_position(
        "Postdoc on Hi-C <in neurons> & more",
        job_title="Postdoc in 3D genomics", hiring_organization="EMBL",
        application_deadline="2026-12-01", application_url="https://embl.org/job?a=1&b=2",
    ))
    text = digest.format_position(pos)
    assert text.startswith("⭐ #Postdoc #Germany #Genome3D #Neuro")
    assert "<b>Postdoc in 3D genomics — EMBL</b>" in text
    assert "&lt;in neurons&gt; &amp; more" in text
    assert "⏰ Deadline: 2026-12-01" in text
    assert 'href="https://embl.org/job?a=1&amp;b=2"' in text


def test_build_messages_sections_and_size_limit():
    positions = []
    for i in range(40):
        positions.append(matching.evaluate(make_position(
            f"Vacancy number {i}: " + "chromatin " * 60, uri=f"at://f{i}")))
    for i in range(5):
        positions.append(matching.evaluate(make_position(
            f"Plain biology PhD number {i} in plant science", uri=f"at://p{i}",
            position_type=["PhD Student"])))
    messages = digest.build_messages(matching.rank(positions), title="TITLE")

    assert len(messages) > 1
    assert all(len(text) <= digest.TELEGRAM_MAX_LENGTH for text, _ in messages)
    assert messages[0][0].startswith("TITLE\n\n⭐ <b>In focus</b>")
    joined = "\n".join(text for text, _ in messages)
    assert joined.count("In focus") == 1
    assert joined.count("PhD &amp; other") == 1
    assert sum(len(group) for _, group in messages) == 45


def test_deliver_reports_undelivered_on_failure(monkeypatch):
    positions = [matching.evaluate(make_position(
        f"Vacancy {i}: " + "chromatin " * 150, uri=f"at://{i}")) for i in range(12)]
    calls = []

    def fake_send(token, chat_id, text):
        calls.append(text)
        return len(calls) < 2  # second message fails

    monkeypatch.setattr(digest, "send_message", fake_send)
    monkeypatch.setattr(digest.time, "sleep", lambda s: None)
    delivered, remaining = digest.deliver(positions, "t", "c")
    assert delivered and remaining
    assert len(delivered) + len(remaining) == 12
    assert {p["uri"] for p in delivered}.isdisjoint(p["uri"] for p in remaining)


# --- state ------------------------------------------------------------------

def test_state_roundtrip_and_limits(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = BotState(path)
    assert state.is_first_run
    state.add_seen(f"at://{i}" for i in range(MAX_SEEN_URIS + 10))
    state.add_seen(["at://0", None])
    state.advance("2026-11-10T10:00:00.000Z")
    state.advance("2026-11-01T10:00:00.000Z")   # never backwards
    state.pending = [{"uri": "at://p"}]
    state.save()

    loaded = BotState(path)
    assert loaded.last_timestamp == "2026-11-10T10:00:00.000Z"
    assert len(loaded.seen_uris) == MAX_SEEN_URIS
    assert loaded.seen_uris[-1] == "at://0"
    assert loaded.pending == [{"uri": "at://p"}]


def test_state_never_advances_into_the_future(tmp_path):
    state = BotState(tmp_path / "s.json")
    state.advance("2999-01-01T00:00:00.000Z")
    assert state.last_timestamp <= utc_now_iso()


def test_state_survives_corrupt_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert BotState(path).is_first_run


# --- end to end (fake Bluesky, fake LLM, fake Telegram) ---------------------

def make_post(uri, text, created, bio="Chromatin lab"):
    message = f"[Bio: {bio}]\n\n{text}"
    return Post(uri=uri, message=message, url=f"https://bsky.app/{uri[-1]}",
                user_handle="lab.bsky.social", created_at=created, source="bluesky",
                raw_text=text, metadata_text=message)


class FakeClassifier:
    """Marks posts containing 'hiring' as jobs; fails on posts containing 'BOOM'."""

    class llm:
        name = "Fake"

    def classify_post(self, text, metadata_text=None, posted_at=None):
        if "BOOM" in text:
            raise LLMUnavailableError("down")
        is_job = "hiring" in text
        return {
            "is_verified_job": is_job,
            "disciplines": ["Biology"] if is_job else None,
            "country": "UK" if is_job else None,
            "position_type": ["Postdoc"] if is_job else None,
            "job_title": None, "hiring_organization": None,
            "application_url": None, "application_deadline": None,
            "location_text": None,
        }


@pytest.fixture
def bot(monkeypatch, tmp_path):
    """Wire run.main to fakes; returns a helper to run it with given posts."""
    import bluesky_search
    import src.sources.bluesky as bluesky

    sent = []
    holder = {}

    class FakeSource:
        def __init__(self, queries=None, limit=50):
            holder["queries"] = queries

        def fetch_posts(self, since_timestamp=None, existing_uris=None):
            holder["since"] = since_timestamp
            existing = existing_uris or set()
            posts = [p for p in holder["posts"]
                     if p.uri not in existing and p.created_at >= since_timestamp]
            return posts, existing | {p.uri for p in holder["posts"]}

    monkeypatch.setattr(bluesky, "BlueskySource", FakeSource)
    monkeypatch.setattr(bluesky_search, "get_classifier", lambda: FakeClassifier())
    monkeypatch.setattr(digest, "send_message",
                        lambda token, chat, text: sent.append(text) or holder.get("ok", True))
    monkeypatch.setattr(digest.time, "sleep", lambda s: None)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "123")
    monkeypatch.setenv("MYBOT_LLM_DELAY", "0")

    state_path = tmp_path / "state.json"
    archive_path = tmp_path / "positions.csv"

    def run_bot(posts, *extra, ok=True):
        holder["posts"] = posts
        holder["ok"] = ok
        return run.main(["--state", str(state_path), "--archive", str(archive_path), *extra])

    run_bot.sent = sent
    run_bot.holder = holder
    run_bot.state_path = state_path
    run_bot.archive_path = archive_path
    return run_bot


NOW = utc_now_iso()


def test_run_sends_matches_and_is_incremental(bot):
    posts = [
        make_post("at://1", "We are hiring a postdoc in Polycomb biology", NOW),
        make_post("at://2", "Congrats to our new PhD graduate!", NOW),
    ]
    assert bot(posts) == 0
    assert len(bot.sent) == 1
    assert "1 new position" in bot.sent[0] and "Polycomb" in bot.sent[0]
    assert "Congrats" not in bot.sent[0]
    assert "postdoc chromatin" in bot.holder["queries"]
    assert "PhD position" in bot.holder["queries"]

    state = json.loads(bot.state_path.read_text())
    assert state["last_timestamp"] == NOW
    assert set(state["seen_uris"]) == {"at://1", "at://2"}
    assert bot.archive_path.read_text().count("at://") == 0  # archive stores web URLs
    assert "Polycomb" in bot.archive_path.read_text()

    # Second run: nothing new, nothing re-sent.
    assert bot(posts) == 0
    assert len(bot.sent) == 1
    assert bot.holder["since"] == NOW


def test_run_dry_run_changes_nothing(bot, capsys):
    posts = [make_post("at://1", "We are hiring a postdoc in Polycomb biology", NOW)]
    assert bot(posts, "--dry-run") == 0
    assert bot.sent == []
    assert not bot.state_path.exists()
    assert "Polycomb" in capsys.readouterr().out


def test_run_keeps_undelivered_for_next_run(bot):
    posts = [make_post("at://1", "We are hiring a postdoc in Polycomb biology", NOW)]
    assert bot(posts, ok=False) == 1
    state = json.loads(bot.state_path.read_text())
    assert len(state["pending"]) == 1 and state["sent"] == []

    assert bot(posts) == 0            # Telegram is back
    state = json.loads(bot.state_path.read_text())
    assert state["pending"] == [] and len(state["sent"]) == 1
    assert len(bot.sent) == 2


def test_run_llm_outage_resumes_from_last_classified_post(bot):
    earlier = "2020-01-01T00:00:00.000Z"
    later = "2020-01-02T00:00:00.000Z"
    posts = [
        make_post("at://1", "We are hiring a postdoc in Polycomb biology", earlier),
        make_post("at://2", "BOOM hiring a research fellow in neurons", later),
    ]
    assert bot(posts, "--lookback-days", "99999") == 0
    state = json.loads(bot.state_path.read_text())
    assert state["last_timestamp"] == earlier
    assert state["seen_uris"] == ["at://1"]
    assert len(bot.sent) == 1

    posts[1] = make_post("at://2", "We are hiring a research fellow in neurons", later)
    assert bot(posts) == 0
    assert len(bot.sent) == 2 and "neurons" in bot.sent[1]


def test_run_requires_telegram_settings(bot, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert bot([]) == 2


def test_non_focus_vacancies_use_compact_format(monkeypatch):
    plain = matching.evaluate(make_position(
        "Postdoc in plant genetics. " + "Long description. " * 40,
        job_title="Postdoc in plant genetics", hiring_organization="JIC",
        application_deadline="2026-12-01"))
    untitled = matching.evaluate(make_position(
        "We have an opening for a postdoc working on yeast. " * 10, uri="at://u"))

    text = digest.build_messages([plain, untitled])[0][0]
    assert "<b>Postdoc in plant genetics — JIC</b>\n⏰ 2026-12-01 · 🔗 " in text
    assert "Long description" not in text
    assert digest.SEPARATOR not in text
    assert len(digest.format_compact(untitled)) < 400

    monkeypatch.setattr(profile, "COMPACT_NON_FOCUS", False)
    assert "Long description" in digest.build_messages([plain])[0][0]


# --- industry positions and the prompt wrapper ------------------------------

class ScriptedProvider(LLMProvider):
    """Returns a JSON answer; treats posts mentioning a company as industry."""

    name = "Scripted"

    def __init__(self):
        self.prompts = []

    def classify(self, text, prompt):
        self.prompts.append(prompt)
        if '"is_verified_job"' not in prompt:
            return "YES"
        industry = "Genentech" in text
        return json.dumps({
            "is_verified_job": "hiring" in text,
            "disciplines": ["Biology"],
            "country": "USA",
            "position_type": ["Postdoc"],
            "job_title": "Scientist, Epigenomics" if industry else None,
            "hiring_organization": "Genentech" if industry else None,
            "application_url": None, "application_deadline": None,
            "location_text": None,
            "sector": "Industry" if industry else "Academia",
        })


def test_wrapper_extends_only_the_metadata_prompt():
    provider = ScriptedProvider()
    classifier = JobClassifier(PersonalLLM(provider))
    result = classifier.classify_post("Genentech is hiring a scientist in epigenomics")

    prompt = provider.prompts[-1]
    assert "ADDITIONAL RULES" in prompt and "Industry research positions" in prompt
    assert prompt.rstrip().endswith("Return ONLY the JSON object, no other text.")
    assert prompt.index("ADDITIONAL RULES") < prompt.index("Return ONLY the JSON object")
    assert "{disciplines}" not in prompt and "{{" not in prompt
    # The upstream result shape is unchanged; the sector travels separately.
    assert "sector" not in result
    assert result["is_verified_job"] is True
    assert classifier.llm.last_sector == "Industry"

    classifier.is_real_job("some text")
    assert "ADDITIONAL RULES" not in provider.prompts[-1]


def test_wrapper_without_industry(monkeypatch):
    monkeypatch.setattr(profile, "INCLUDE_INDUSTRY", False)
    provider = ScriptedProvider()
    JobClassifier(PersonalLLM(provider)).classify_post("We are hiring a postdoc")
    assert "Industry research positions" not in provider.prompts[-1]
    assert "staff scientist" in provider.prompts[-1]


@pytest.mark.parametrize("response,expected", [
    ('{"sector": "Industry"}', "Industry"),
    ('```json\n{"sector": "industry "}\n```', "Industry"),
    ('[{"sector": "Industry"}]', "Industry"),
    ('{"sector": "Academia"}', "Academia"),
    ('{"is_verified_job": true}', "Academia"),
    ("not json", "Academia"),
    ('"Industry"', "Academia"),
])
def test_parse_sector(response, expected):
    assert parse_sector(response) == expected


def test_industry_positions_are_tagged_and_optional(monkeypatch):
    pos = make_position("Scientist II, cell biology assays, biotech in Basel",
                        sector="Industry", job_title="Scientist II", hiring_organization="Roche")
    result = matching.evaluate(pos)
    assert result["is_industry"] is True and result["is_priority_level"] is True
    assert digest.format_compact(result).startswith("#Industry #Scientist #Germany #CellBio")

    text = digest.build_messages([result])[0][0]
    assert text.startswith("🏢 <b>Industry R&amp;D</b>")

    focus = matching.evaluate(make_position("Scientist, chromatin biology", sector="Industry"))
    assert digest.format_position(focus).startswith("⭐ #Industry #Scientist #Germany #Epigenetics")
    assert digest.build_messages([focus])[0][0].startswith("⭐ <b>In focus</b>")

    monkeypatch.setattr(profile, "INCLUDE_INDUSTRY", False)
    assert matching.evaluate(pos) is None


def test_run_marks_industry_positions(bot, monkeypatch):
    import bluesky_search
    monkeypatch.setattr(bluesky_search, "get_classifier",
                        lambda: JobClassifier(ScriptedProvider()))
    posts = [
        make_post("at://1", "Genentech is hiring a scientist in epigenomics", NOW, bio="Recruiter"),
        make_post("at://2", "We are hiring a postdoc in plant genetics", NOW, bio="Plant lab"),
    ]
    assert bot(posts) == 0
    text = bot.sent[0]
    assert "#Industry #Scientist #USA #Epigenetics" in text
    assert "Scientist, Epigenomics — Genentech" in text
    assert "#Postdoc #USA" in text
    rows = bot.archive_path.read_text()
    assert "Industry" in rows and "Academia" in rows


def test_time_budget_stops_early_and_resumes(bot, monkeypatch):
    clock = iter(range(0, 100000, 400))          # every check is 400 "seconds" later
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    earlier = "2020-01-01T00:00:00.000Z"
    later = "2020-01-02T00:00:00.000Z"
    posts = [
        make_post("at://1", "We are hiring a postdoc in Polycomb biology", earlier),
        make_post("at://2", "We are hiring a research fellow in neurons", later),
    ]
    # Budget of 10 min = 600 s: first post at 400 s passes, second at 800 s does not.
    assert bot(posts, "--lookback-days", "99999", "--max-minutes", "10") == 0
    state = json.loads(bot.state_path.read_text())
    assert state["last_timestamp"] == earlier and state["seen_uris"] == ["at://1"]
    assert len(bot.sent) == 1

    assert bot(posts, "--max-minutes", "0") == 0
    assert len(bot.sent) == 2 and "neurons" in bot.sent[1]
