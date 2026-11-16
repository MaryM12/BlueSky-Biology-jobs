"""Search profile for the personal job bot. EDIT THIS FILE to tune the bot.

Three things are configured here:

1. EXTRA_QUERIES    - additional Bluesky searches (on top of the upstream ones);
                      INDUSTRY_QUERIES are added when INCLUDE_INDUSTRY is on
2. FOCUS_TOPICS     - "my" research areas; matching vacancies get a star and go first
3. WET_LAB_TOPICS   - broader wet-lab biology tags (molecular / cell biology)

Patterns are regular expressions. They are matched case-insensitively against
the post text, the link preview and the author's bio. Prefix a pattern with
``(?-i:...)`` to make it case-sensitive (used for short acronyms like TAD).
"""

# --------------------------------------------------------------------------
# 1. Search queries
# --------------------------------------------------------------------------
# Bluesky search matches posts that contain all words of a query, so short
# two/three-word queries work best. Each query returns the newest posts.

# Set to False to search ONLY with the queries below (skips the generic
# upstream list such as "PhD position", "postdoc position", ...).
USE_UPSTREAM_QUERIES = True

EXTRA_QUERIES = [
    # --- post-PhD research positions (any field; the filter narrows them) ---
    "postdoctoral position",
    "postdoctoral researcher",
    "postdoc opening",
    "postdoc opportunity",
    "research fellow position",
    "research associate position",
    "research scientist position",
    "staff scientist position",
    "hiring research fellow",
    "join our lab postdoc",
    # --- wet-lab biology ---
    "postdoc molecular biology",
    "postdoc cell biology",
    "postdoc developmental biology",
    "postdoc stem cells",
    "postdoc genomics",
    "postdoc wet lab",
    "PhD molecular biology",
    "PhD cell biology",
    "position molecular biology",
    "position cell biology",
    "research assistant molecular biology",
    # --- focus: epigenetics, chromatin, 3D genome ---
    "postdoc epigenetics",
    "postdoc chromatin",
    "postdoc Hi-C",
    "postdoc 3D genome",
    "postdoc genome organization",
    "postdoc nuclear organization",
    "postdoc Polycomb",
    "postdoc gene regulation",
    "postdoc DNA methylation",
    "postdoc transcription",
    "postdoc single-cell",
    "position epigenetics",
    "position chromatin",
    "position 3D genome",
    "hiring epigenetics",
    "hiring chromatin",
    "PhD epigenetics",
    "PhD chromatin",
    # --- focus: neurons / brain ---
    "postdoc neuroscience",
    "postdoc neurobiology",
    "postdoc neurons",
    "postdoc neuroepigenetics",
    "postdoc brain organoids",
    "position neurobiology",
]

# Industry R&D positions (scientist / research associate roles at biotech and
# pharma companies). Set to False to keep the bot academic-only: the queries
# below are skipped and company vacancies are dropped.
INCLUDE_INDUSTRY = True

INDUSTRY_QUERIES = [
    "biotech hiring scientist",
    "scientist position biotech",
    "pharma scientist position",
    "industry postdoc",
    "research scientist genomics",
    "senior scientist genomics",
    "scientist epigenetics",
    "scientist chromatin",
    "scientist molecular biology",
    "scientist cell biology",
    "scientist neuroscience hiring",
    "research associate biotech",
    "R&D scientist biology",
]

# --------------------------------------------------------------------------
# 2. Focus topics (starred, shown first). tag -> list of regex patterns
# --------------------------------------------------------------------------
FOCUS_TOPICS = {
    "Epigenetics": [
        r"epigen(etic|om)",
        r"chromatin",
        r"histone",
        r"polycomb",
        r"(?-i:\bPRC[12]\b)",
        r"dna methylation",
        r"epiallel",
        r"methylom",
        r"nucleosome",
        r"\bchip-?seq\b",
        r"\batac-?seq\b",
        r"cut\s?&\s?(run|tag)",
        r"gene regulat",
        r"transcriptional regulat",
        r"\benhancers?\b",
    ],
    "Genome3D": [
        r"\bhi-?c\b",
        r"\bmicro-?c\b",
        r"\b[34]d[- ](genome|genomics|nucleome|chromatin)",
        r"genome (organi[sz]ation|architecture|folding|topology)",
        r"chromosome (conformation|organi[sz]ation|folding)",
        r"nuclear (organi[sz]ation|architecture)",
        r"(?-i:\bTADs?\b)",
        r"loop extrusion",
        r"\bcohesin\b",
        r"(?-i:\bCTCF\b)",
    ],
    "Neuro": [
        r"\bneurons?\b",
        r"\bneuronal\b",
        r"neurobiolog",
        r"neurogenesis",
        r"neuroepigen",
        r"neurogenomic",
        r"(brain|cortical|cerebral|neural) organoid",
        r"neural (stem|progenitor)",
        r"\bglia(l)?\b",
        r"astrocyte",
        r"microglia",
        r"oligodendrocyte",
        r"\bsynap(se|ses|tic)\b",
        r"(molecular|cellular) neuroscience",
    ],
}

# --------------------------------------------------------------------------
# 3. Wet-lab biology tags (included, not starred). tag -> regex patterns
# --------------------------------------------------------------------------
WET_LAB_TOPICS = {
    "MolBio": [
        r"molecular (biolog|genetic)",
        r"biochemi",
        r"\bcrispr\b",
        r"genome editing",
        r"rna biolog",
        r"wet[- ]?lab",
    ],
    "CellBio": [
        r"cell(ular)? biolog",
        r"stem cell",
        r"organoid",
        r"cell culture",
        r"live[- ]cell",
        r"organelle",
        r"cytoskelet",
        r"developmental biolog",
    ],
}

# --------------------------------------------------------------------------
# 4. Pre-filter: which posts are worth showing to the LLM at all
# --------------------------------------------------------------------------
# Every LLM call costs time and free-tier quota, and most "PhD position" posts
# are about other fields. With the pre-filter on, a post goes to the LLM only
# if its text, link preview or author bio mentions molecular / cellular life
# science (the patterns below) or any topic pattern above. Ecology, field
# biology and non-biology posts are skipped. Set to False to classify every
# post (needs a generous LLM quota, e.g. Mistral).
PREFILTER_ENABLED = True

LIFE_SCIENCE_PATTERNS = [
    # "biology" alone is too broad (ecology, zoology...), so name the kinds.
    r"\bbio(chem|phys|med|info|tech|eng|mol)",
    r"(chemical|synthetic|systems|structural|computational|quantitative|cancer) biolog",
    r"\bcell(s|ular)?\b", r"molecul", r"\bgen(e|es|etics?|omes?|omics?)\b", r"genom",
    r"neuro", r"\bbrain", r"protein", r"proteosta", r"\bm?i?rnas?\b", r"\bdna\b",
    r"chromatin", r"chromosom", r"epigen", r"transcript", r"microb", r"immun",
    r"cancer", r"tumou?r", r"oncolog", r"\bstem\b", r"organoid", r"crispr",
    r"sequencing", r"\b(multi[- ]?)?omics\b", r"\bvir(us|uses|al|olog)", r"bacteri", r"pathogen",
    r"zebrafish", r"drosophila", r"\bmouse\b", r"\bmice\b", r"yeast", r"elegans",
    r"embryo", r"tissue", r"extracellular", r"metabol", r"mitochond", r"membrane",
    r"enzym", r"cryo-?em", r"microscop", r"synap", r"developmental", r"life scien",
    r"wet[- ]?lab", r"single[- ]cell", r"proteom", r"lipid", r"antibod", r"nanobod",
    r"vaccin", r"regenerat", r"signal(l)?ing", r"mutat", r"pharmac", r"physiolog",
    r"\b(emb[lo]|irb|crg|imba|max planck|crick|sanger|janelia|igbmc)\b",
]

# Posts about these fields are skipped - but only when nothing in the post
# points to molecular or cellular work: a focus or wet-lab topic above, or any
# of MOLECULAR_PATTERNS below, keeps the post. So "single-cell atlas of
# environmental exposure" or "plant microbiome" stay, while "forest ecology"
# or "cognitive neuroscience" go.
MOLECULAR_PATTERNS = [
    r"molecul", r"\bcell(s|ular)?\b", r"protein", r"\bdna\b", r"\bm?i?rnas?\b",
    r"\bgen(e|es|etics?|omes?|omics?)\b", r"genom", r"chromatin", r"chromosom",
    r"epigen", r"transcript", r"crispr", r"sequencing", r"\b(multi[- ]?)?omics\b",
    r"proteom", r"biochem", r"biophys", r"microscop", r"cryo-?em", r"mitochond",
    r"organoid", r"\bstem\b", r"enzym", r"single[- ]cell", r"microb", r"bacteri",
    r"\bvir(us|uses|al|olog)", r"immun", r"cancer", r"tumou?r", r"antibod",
    r"zebrafish", r"drosophila", r"\bmouse\b", r"\bmice\b", r"yeast", r"elegans",
]

OFF_TOPIC_PATTERNS = [
    r"ecolog", r"biodiversity", r"conservation", r"wildlife", r"fisher", r"forestry",
    r"agricultur", r"agro", r"\bsoil", r"\bcrop", r"environmental", r"contaminant",
    r"archaeolog", r"morphometric", r"paleo", r"palaeo",
    r"cognitive", r"behaviou?ral", r"psycholog", r"\bfmri\b", r"psychiatr",
    r"clinical trial", r"nursing", r"pa?ediatric", r"epidemiolog", r"public health",
]

# --------------------------------------------------------------------------
# 5. Bioinformatics vs wet lab
# --------------------------------------------------------------------------
# A vacancy is shown as bioinformatics when it uses computational wording and
# no hands-on wording (mixed wet/dry positions stay with the wet lab).
COMPUTATIONAL_PATTERNS = [
    r"bioinformati", r"computational", r"machine learning", r"deep learning",
    r"data scien", r"biostatist", r"\balgorithm", r"software", r"in silico",
    r"modell?ing", r"(?-i:\bAI\b)", r"(?-i:\bML\b)", r"\bpipelines?\b",
]
HANDS_ON_PATTERNS = [
    r"wet[- ]?lab", r"experimental", r"\bbench\b", r"hands-on", r"\bcloning\b",
    r"cell culture", r"microscop", r"mouse models?", r"\bassays?\b",
]

# --------------------------------------------------------------------------
# 6. PhD / Master positions
# --------------------------------------------------------------------------
#   "channel" - sent to a separate Telegram channel (secret
#               TELEGRAM_CHANNEL_ID_PHD); until that secret is set they are
#               not sent and not even shown to the LLM (saves quota)
#   "main"    - sent to the main chat as the last section
#   "off"     - never sent
PHD_POSITIONS = "channel"

# Order in which posts are shown to the LLM when its quota or time is short:
#   0 - mentions a focus topic            (kept for the next run if not reached)
#   1 - research-staff wording + life science   (kept for the next run)
#   2 - other life-science posts, e.g. PhD positions (dropped if not reached)
# Within a group the newest posts go first.
STAFF_ROLE_PATTERNS = [
    r"post-?doc", r"research fellow", r"research associate", r"research assistant",
    r"\bscientist\b", r"\bresearcher\b",
]
QUEUE_MAX = 300           # at most this many posts wait for the next run
QUEUE_MAX_AGE_DAYS = 7    # ...and none older than this

# Faculty-level openings are not what this bot is for. A post is skipped when
# it uses this wording and mentions no postdoc / PhD / research-staff role;
# a vacancy is dropped when its extracted title matches.
EXCLUDED_ROLE_PATTERNS = [
    r"(assistant|associate|full) professor", r"tenure[- ]track", r"professorship",
    r"faculty (position|search|opening|job)", r"\blecturer\b", r"group leader",
    r"department (chair|head)",
]

# Vacancies whose application deadline has already passed are dropped.
DROP_EXPIRED_DEADLINES = True

# Groq's free plan has a small daily token quota per model. When the main
# model's quota is used up, the bot continues with these models (same key).
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b"]

# --------------------------------------------------------------------------
# What gets through, and in which order
# --------------------------------------------------------------------------
# A verified vacancy is sent when the LLM labels it with one of these
# disciplines OR when it matches any topic pattern above.
RELEVANT_DISCIPLINES = {"Biology"}

# Position types treated as "research staff" (shown before PhD/Master ones).
# The LLM labels research fellow / research associate / staff scientist and
# every role that requires a PhD as "Postdoc".
PRIORITY_POSITION_TYPES = {"Postdoc", "Research Assistant"}

# Position types to drop entirely, e.g. {"Master Student"}. Empty = keep all.
EXCLUDED_POSITION_TYPES: set[str] = set()

# Focus vacancies always get a full card with a text excerpt. Everything else
# is shown as a short three-line entry to keep the digest readable; set to
# False to get full cards for all vacancies.
COMPACT_NON_FOCUS = True

# First run only: how far back to look (days). Later runs are incremental.
FIRST_RUN_LOOKBACK_DAYS = 14
