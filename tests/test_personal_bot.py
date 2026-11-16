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
        application_deadline="2999-12-01", application_url="https://embl.org/job?a=1&b=2",
    ))
    text = digest.format_position(pos)
    assert text.startswith("⭐ #Postdoc #Germany #Genome3D #Neuro")
    assert "<b>Postdoc in 3D genomics — EMBL</b>" in text
    assert "&lt;in neurons&gt; &amp; more" in text
    assert "⏰ Deadline: 2999-12-01" in text
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
    """Marks posts containing 'hiring' as jobs; fails while ``down`` is set."""

    class llm:
        name = "Fake"

    down = False

    def classify_post(self, text, metadata_text=None, posted_at=None):
        if FakeClassifier.down:
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
    monkeypatch.setattr(FakeClassifier, "down", False)

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



def test_run_requires_telegram_settings(bot, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert bot([]) == 2



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



# --- pre-filter and Groq fallbacks ------------------------------------------

@pytest.mark.parametrize("text,bio,expected", [
    ("Postdoc in chromatin biology", "", True),
    ("PhD position on CRISPR screens in mouse embryos", "", True),
    ("We are hiring a postdoc! Link below.", "Cell biologist studying mitochondria", True),
    ("Scientist II, antibody engineering", "", True),
    ("PhD position in medieval history", "Historian", False),
    ("Funded PhD on beavers and lowland ecosystems, lots of fieldwork", "", False),
    ("Postdoc in labour economics", "Economist, coffee drinker", False),
])
def test_prefilter(text, bio, expected):
    assert matching.passes_prefilter(make_position(text, bio=bio)) is expected


def test_prefilter_can_be_disabled(monkeypatch):
    pos = make_position("PhD position in medieval history")
    monkeypatch.setattr(profile, "PREFILTER_ENABLED", False)
    assert matching.passes_prefilter(pos) is True


def test_every_topic_match_passes_the_prefilter(monkeypatch):
    monkeypatch.setattr(profile, "LIFE_SCIENCE_PATTERNS", [])
    assert matching.passes_prefilter(make_position("how TADs form")) is True


def test_run_skips_llm_for_prefiltered_posts(bot, monkeypatch):
    import bluesky_search
    provider = ScriptedProvider()
    monkeypatch.setattr(bluesky_search, "get_classifier", lambda: JobClassifier(provider))
    posts = [
        make_post("at://1", "We are hiring a historian of medieval trade", NOW, bio="History dept"),
        make_post("at://2", "We are hiring a postdoc in chromatin biology", NOW, bio="Lab"),
    ]
    assert bot(posts) == 0
    assert len(provider.prompts) == 1                       # one LLM call only
    state = json.loads(bot.state_path.read_text())
    assert set(state["seen_uris"]) == {"at://1", "at://2"}  # both are done
    assert len(bot.sent) == 1 and "chromatin" in bot.sent[0]


def test_max_classify_counts_llm_calls_not_posts(bot, monkeypatch):
    posts = [make_post(f"at://h{i}", f"We are hiring a historian number {i}",
                       f"2020-01-0{i + 1}T00:00:00.000Z", bio="History") for i in range(3)]
    posts.append(make_post("at://b", "We are hiring a postdoc in chromatin biology",
                           "2020-01-05T00:00:00.000Z"))
    assert bot(posts, "--lookback-days", "99999", "--max-classify", "1") == 0
    assert len(bot.sent) == 1


def test_groq_fallback_chain_and_delay(monkeypatch):
    from src.llm.fallback import FallbackProvider

    class Holder:
        llm = ScriptedProvider()

    for key in ("MISTRAL_API_KEY", "GEMINI_API_KEY", "NVIDIA_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    holder = Holder()
    run.add_groq_fallbacks(holder)
    assert isinstance(holder.llm, ScriptedProvider)         # no Groq key: unchanged
    assert run.default_llm_delay() == 1.0

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(profile, "GROQ_FALLBACK_MODELS", ["model-a", "model-b"])
    run.add_groq_fallbacks(holder)
    assert isinstance(holder.llm, FallbackProvider)
    assert holder.llm.fallback.model == "model-b"
    assert holder.llm.primary.fallback.model == "model-a"
    assert "model-a" in holder.llm.name and "model-b" in holder.llm.name
    assert run.default_llm_delay() == 7.0

    monkeypatch.setenv("MISTRAL_API_KEY", "x")
    assert run.default_llm_delay() == 1.0


# --- priority order, queue and exclusions -----------------------------------

def days_ago(days):
    from datetime import datetime, timedelta, timezone
    moment = datetime.now(timezone.utc) - timedelta(days=days)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.mark.parametrize("text,bio,expected", [
    ("PhD position on Polycomb and chromatin loops", "", 0),
    ("We are hiring a postdoc!", "3D genome lab", 0),
    ("Postdoc position in yeast metabolism", "", 1),
    ("Research associate, antibody engineering", "", 1),
    ("PhD position in yeast metabolism", "", 2),
    ("PhD position in medieval history", "", None),
    ("Tenure-track Assistant Professor in Cell Biology, apply now", "", None),
    ("Assistant Professor and two postdoc positions in cell biology", "", 1),
])
def test_triage(text, bio, expected):
    assert matching.triage(make_position(text, bio=bio)) == expected


def test_triage_without_prefilter(monkeypatch):
    monkeypatch.setattr(profile, "PREFILTER_ENABLED", False)
    assert matching.triage(make_position("PhD position in medieval history")) == 3


def test_faculty_titles_and_expired_deadlines_are_dropped(monkeypatch):
    faculty = make_position("Join us! Postdocs welcome too. Cell biology.",
                            job_title="Tenure-Track Assistant Professor in Neurobiology")
    assert matching.evaluate(faculty) is None

    expired = make_position("Postdoc in cell biology", application_deadline="2020-01-31")
    upcoming = make_position("Postdoc in cell biology", application_deadline="2999-01-31")
    unclear = make_position("Postdoc in cell biology", application_deadline="rolling")
    assert matching.evaluate(expired) is None
    assert matching.evaluate(upcoming) is not None
    assert matching.evaluate(unclear) is not None

    monkeypatch.setattr(profile, "DROP_EXPIRED_DEADLINES", False)
    assert matching.evaluate(expired) is not None


def test_compact_entries_show_title_and_snippet(monkeypatch):
    plain = matching.evaluate(make_position(
        "Postdoc in plant genetics. " + "Long description. " * 40,
        job_title="Postdoc in plant genetics", hiring_organization="JIC",
        application_deadline="2999-12-01"))
    untitled = matching.evaluate(make_position(
        "We have an opening for a postdoc working on yeast. " * 10, uri="at://u"))

    text = digest.build_messages([plain, untitled])[0][0]
    assert "<b>Postdoc in plant genetics — JIC</b>\nPostdoc in plant genetics. Long" in text
    assert "…\n⏰ 2999-12-01 · 🔗 " in text
    assert digest.SEPARATOR not in text
    assert len(digest.format_compact(plain)) < 450
    assert len(digest.format_compact(untitled)) < 400

    monkeypatch.setattr(profile, "COMPACT_NON_FOCUS", False)
    assert digest.SNIPPET_LENGTH > digest.COMPACT_SNIPPET_LENGTH
    assert len(digest.build_messages([plain])[0][0]) > 500


def test_candidates_are_ordered_by_tier_then_newest():
    posts = [
        make_post("at://phd-new", "PhD position in yeast metabolism", days_ago(0), bio="x"),
        make_post("at://staff-old", "Postdoc position in yeast metabolism", days_ago(3), bio="x"),
        make_post("at://staff-new", "Postdoc position in mouse immunology", days_ago(1), bio="x"),
        make_post("at://focus-old", "PhD position on chromatin loops", days_ago(5), bio="x"),
        make_post("at://skip", "PhD position in medieval history", days_ago(0), bio="x"),
    ]
    candidates, skipped = run.build_candidates([], posts)
    assert skipped == 1
    assert [data["uri"] for _, data in candidates] == [
        "at://focus-old", "at://staff-new", "at://staff-old", "at://phd-new",
    ]
    assert [tier for tier, _ in candidates] == [0, 1, 1, 2]


def test_queue_keeps_only_recent_top_priority_posts(monkeypatch):
    leftovers = [
        (0, {"uri": "focus", "created": days_ago(1)}),
        (0, {"uri": "focus-too-old", "created": days_ago(30)}),
        (1, {"uri": "staff", "created": days_ago(2)}),
        (2, {"uri": "phd", "created": days_ago(0)}),
    ]
    assert [d["uri"] for d in run.next_queue(leftovers)] == ["focus", "staff"]
    monkeypatch.setattr(profile, "QUEUE_MAX", 1)
    assert [d["uri"] for d in run.next_queue(leftovers)] == ["focus"]


def test_run_llm_outage_queues_top_priority_posts(bot, monkeypatch):
    posts = [
        make_post("at://1", "We are hiring a postdoc in Polycomb biology", days_ago(1)),
        make_post("at://2", "We are hiring a PhD student in yeast metabolism", days_ago(0), bio="x"),
    ]
    monkeypatch.setattr(FakeClassifier, "down", True)
    assert bot(posts) == 0
    state = json.loads(bot.state_path.read_text())
    assert bot.sent == []
    assert [p["uri"] for p in state["queue"]] == ["at://1"]      # focus post waits
    assert set(state["seen_uris"]) == {"at://1", "at://2"}      # the PhD one is dropped

    monkeypatch.setattr(FakeClassifier, "down", False)
    assert bot(posts) == 0                                        # nothing new to fetch
    state = json.loads(bot.state_path.read_text())
    assert state["queue"] == []
    assert len(bot.sent) == 1 and "Polycomb" in bot.sent[0]


def test_time_budget_classifies_best_posts_first(bot, monkeypatch):
    clock = iter(range(0, 100000, 400))          # every check is 400 "seconds" later
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    posts = [
        make_post("at://phd", "We are hiring a PhD student in yeast metabolism", days_ago(0), bio="x"),
        make_post("at://old", "We are hiring a postdoc in Polycomb biology", days_ago(2)),
        make_post("at://new", "We are hiring a research fellow in neurons", days_ago(1)),
    ]
    # Budget of 10 min = 600 s: the first candidate (at 400 s) passes, the next does not.
    assert bot(posts, "--max-minutes", "10") == 0
    assert len(bot.sent) == 1 and "neurons" in bot.sent[0]
    state = json.loads(bot.state_path.read_text())
    assert [p["uri"] for p in state["queue"]] == ["at://old"]

    assert bot(posts, "--max-minutes", "0") == 0
    assert len(bot.sent) == 2 and "Polycomb" in bot.sent[1]


def test_dry_run_prints_plain_text(bot, capsys):
    posts = [make_post("at://1", "We are hiring a postdoc in Polycomb & chromatin", NOW)]
    assert bot(posts, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "Polycomb & chromatin" in out and "&amp;" not in out
