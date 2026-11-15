"""LLM wrapper that adapts the upstream classification prompt to the profile.

The upstream classifier and its prompt stay untouched (so the fork can be
synced without conflicts). This wrapper sits between the classifier and the
real provider: it appends a few extra rules to the metadata prompt and reads
one extra field ("sector") from the model's answer.
"""

import json
import re

from src.llm.base import LLMProvider

from . import profile

_FINAL_LINE = "Return ONLY the JSON object"

BASE_RULES = (
    "- Research fellow, research associate, research scientist and staff scientist "
    "positions are open research positions: is_verified_job true, position_type \"Postdoc\".\n"
    "- Molecular, cellular or developmental neuroscience / neurobiology (neurons, glia, "
    "synapses, brain organoids, neurogenomics) is Biology; add Psychology or Medicine "
    "only if the post is also behavioural/cognitive or clinical.\n"
)

INDUSTRY_RULES = (
    "- Industry research positions count as well: scientist, research scientist, R&D "
    "scientist, research associate and industry-postdoc roles at biotech, pharma or other "
    "companies are is_verified_job true. Use position_type \"Postdoc\" when the role is "
    "scientist-level or requires a PhD, otherwise \"Research Assistant\".\n"
    "- Company jobs that are not hands-on research stay false: sales, marketing, business "
    "development, regulatory, HR, software engineering without research, and executive "
    "roles (director, VP, head of).\n"
)

SECTOR_RULE = (
    "- Add one more field to the JSON object: \"sector\". Use \"Industry\" when the employer "
    "is a company (biotech, pharma, startup, CRO); otherwise \"Academia\" (universities, "
    "research institutes, hospitals, government labs).\n"
)


def extra_rules() -> str:
    rules = BASE_RULES
    if profile.INCLUDE_INDUSTRY:
        rules += INDUSTRY_RULES
    return (
        "ADDITIONAL RULES (they take precedence over the rules above):\n"
        + rules + SECTOR_RULE + "\n"
    )


def parse_sector(response: str) -> str:
    """Read the sector from a raw model answer; anything unclear is Academia."""
    text = re.sub(r"^```(?:json)?\s*", "", response.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return "Academia"
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if isinstance(data, dict) and str(data.get("sector", "")).strip().lower().startswith("indus"):
        return "Industry"
    return "Academia"


class PersonalLLM(LLMProvider):
    """Wraps a provider; extends metadata prompts and remembers the last sector."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.last_sector = "Academia"

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", type(self.inner).__name__)

    def classify(self, text: str, prompt: str) -> str:
        is_metadata_prompt = '"is_verified_job"' in prompt
        if is_metadata_prompt:
            head, sep, tail = prompt.rpartition(_FINAL_LINE)
            if sep:
                prompt = head + extra_rules() + sep + tail
            else:
                prompt = prompt + "\n\n" + extra_rules()
        response = self.inner.classify(text, prompt)
        if is_metadata_prompt:
            self.last_sector = parse_sector(response)
        return response
