"""
Shared text analysis for building pairs and reviewing data.

Pronoun counting is only a valid label when the occupation-holder is the
ONLY person the pronouns can refer to (README, lesson #9). So on top of the
pronoun filters, `analyze()` identifies the character's first name and
rejects texts with more than one named person.

Names are detected with the Kantrowitz names corpus (resources/names, see
its README for license/credit) restricted to proper nouns via spaCy POS
tags. spaCy NER alone was tested and rejected: it tagged "Marcus" as a
place and missed "Priya".
"""

import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMES_DIR = REPO_ROOT / "resources" / "names"

MALE_PRONOUNS = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
FEMALE_PRONOUNS = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)

# Safety net for real, well-known people (observed: Mark Zuckerberg for
# "CEO"). Not exhaustive -- extend with names found in manual review.
REAL_PERSON_BLOCKLIST = [
    "mark zuckerberg", "elon musk", "bill gates", "jeff bezos",
    "tim cook", "sundar pichai", "satya nadella", "sam altman",
    "larry page", "sergey brin", "jack ma", "warren buffett",
    "steve jobs", "richard branson", "indra nooyi", "mary barra",
    "ginni rometty", "sheryl sandberg",
    "florence nightingale", "marie curie", "albert einstein",
    "amelia earhart", "neil armstrong",
]

REFUSAL_PATTERN = re.compile(
    r"\b(as an ai|ai language model|i'm sorry|i am sorry|i cannot|i can't "
    r"(help|assist|write)|i'm unable|i am unable)\b",
    re.IGNORECASE,
)

# Dialogue is stripped before checking for first-person narration.
QUOTED = re.compile(r'"[^"]*"|“[^”]*”')
FIRST_PERSON = re.compile(r"\bI\b|\b(?i:my|me|myself)\b")

# Other-person nouns: flagged in review, NOT excluded (a nurse's text may
# mention "patients" harmlessly). Whether they cause misattribution in the
# short set is measured by the manual audit, and tightened if needed.
OTHER_PERSON = re.compile(
    r"\b(colleague|coworker|co-worker|boss|manager|assistant|friend|patient|"
    r"customer|client|visitor|guest|student|child|kid|son|daughter|wife|"
    r"husband|partner|mother|father|mom|dad|sister|brother|passenger|"
    r"man|woman|boy|girl|stranger)s?\b",
    re.IGNORECASE,
)


def pronoun_counts(text):
    return {
        "male": len(MALE_PRONOUNS.findall(text)),
        "female": len(FEMALE_PRONOUNS.findall(text)),
    }


def pronoun_gender(text):
    """'male' / 'female' (pure), 'mixed', or 'none'."""
    c = pronoun_counts(text)
    if c["male"] and c["female"]:
        return "mixed"
    if c["male"]:
        return "male"
    if c["female"]:
        return "female"
    return "none"


def mentions_real_person(text):
    lowered = text.lower()
    return any(name in lowered for name in REAL_PERSON_BLOCKLIST)


def has_first_person_narration(text):
    return bool(FIRST_PERSON.search(QUOTED.sub(" ", text)))


def exclusion_reason(record, min_pronoun_count):
    """Text-level reasons a completion can't be used at all, or None."""
    text = record["completion"]
    if not text.strip():
        return "empty"
    if record.get("truncated", False):
        return "truncated"
    if mentions_real_person(text):
        return "real_person"
    if REFUSAL_PATTERN.search(text):
        return "refusal"
    if has_first_person_narration(text):
        return "first_person"
    c = pronoun_counts(text)
    if c["male"] + c["female"] < min_pronoun_count:
        return "no_signal"
    if c["male"] and c["female"]:
        return "mixed"
    return None


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def name_lists():
    """(male_only, female_only, unisex) sets of first names."""
    male = set(open(NAMES_DIR / "male.txt").read().split())
    female = set(open(NAMES_DIR / "female.txt").read().split())
    return frozenset(male - female), frozenset(female - male), frozenset(male & female)


def name_gender(name):
    """'male', 'female', 'unisex' or None (not in the corpus)."""
    male, female, unisex = name_lists()
    if name in male:
        return "male"
    if name in female:
        return "female"
    if name in unisex:
        return "unisex"
    return None


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


TITLES = {"Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Sir", "Madam", "Mx"}


# Corpus names that are also everyday words: accepted as names only when
# spaCy NER tags them PERSON. The PROPN tag is not enough: sentence-initial
# "Hope filled the room" is tagged PROPN (observed).
AMBIGUOUS_NAMES = {
    "will", "may", "hope", "grace", "joy", "faith", "rose", "dawn", "summer",
    "page", "bill", "mark", "art", "sue", "pat", "jack", "guy", "earl",
    "june", "april", "august", "ray", "dean", "sky", "sunny", "autumn",
    "winter", "holly", "ivy", "iris", "lily", "violet", "hazel", "ruby",
    "amber", "crystal", "pearl", "ginger", "honey", "angel", "chase",
    "hunter", "miles", "norm", "rich", "gay", "constance", "prudence",
    "patience", "liberty", "destiny", "harmony", "melody", "story", "lee",
    "ash", "reed", "day", "love", "star", "brook", "river", "stone", "cliff",
    "don", "frank", "grant", "gene", "cole", "wade", "drew", "rob", "sterling",
}
NON_NAME_POS = {"VERB", "AUX", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT", "NUM"}


def _is_name_part(tok):
    """
    Part of a person reference. PROPN alone is not enough: spaCy tags
    sentence-initial "Emily" as an adverb (observed), so a capitalized
    corpus name counts unless its POS rules it out or it is an everyday word.
    """
    if tok.lower_ in AMBIGUOUS_NAMES:
        return tok.ent_type_ == "PERSON"
    if tok.pos_ == "PROPN":
        return True
    if not tok.text[:1].isupper() or name_gender(tok.text) is None:
        return False
    return tok.pos_ not in NON_NAME_POS


def _propn_spans(doc):
    """Maximal runs of consecutive name parts ("Dr. Emily Chen")."""
    spans, cur = [], []
    for tok in doc:
        if _is_name_part(tok):
            cur.append(tok)
        elif cur:
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def find_names(doc):
    """
    First names of the people in a spaCy doc, as {name: gender}.

    Each run of consecutive proper nouns is one person reference: its first
    token found in the names corpus (skipping titles and words like
    "Nurse") is the first name; later tokens are surnames and ignored --
    otherwise "Emily Chen" would count as two people ("Chen" is in the
    corpus). A run with no corpus name that spaCy tags as PERSON yields an
    unknown name (gender None).
    """
    person_idx = {t.i for ent in doc.ents if ent.label_ == "PERSON" for t in ent}
    found = {}
    for span in _propn_spans(doc):
        tokens = [t for t in span if t.text.rstrip(".") not in TITLES]
        name = next((t.text for t in tokens if name_gender(t.text) is not None), None)
        if name is not None:
            found[name] = name_gender(name)
        elif tokens and any(t.i in person_idx for t in tokens):
            found.setdefault(tokens[0].text, None)
    return found


def analyze(record, min_pronoun_count, nlp=None):
    """
    Full per-completion analysis. Returns a dict:
        reason  -- exclusion reason or None
        gender  -- pure pronoun gender ('male'/'female') when usable
        name, name_gender -- the single character's first name, if usable
        other_person_flag -- other-person noun present (review only)
    Order matters: cheap text filters first, spaCy only for survivors.
    """
    out = {"reason": None, "gender": None, "name": None, "name_gender": None,
           "other_person_flag": False}
    reason = exclusion_reason(record, min_pronoun_count)
    if reason:
        out["reason"] = reason
        return out
    text = record["completion"]
    out["gender"] = pronoun_gender(text)
    out["other_person_flag"] = bool(OTHER_PERSON.search(text))

    doc = (nlp or get_nlp())(text)
    names = find_names(doc)
    if len(names) == 0:
        out["reason"] = "no_name"
    elif len(names) > 1:
        out["reason"] = "multi_person"
    else:
        (name, g), = names.items()
        out["name"], out["name_gender"] = name, g
        if g is None:
            out["reason"] = "unknown_name"
        elif g != "unisex" and g != out["gender"]:
            # "Sarah ... he": name and pronouns disagree -> either a second
            # person or incoherent text. Either way the label is unreliable.
            out["reason"] = "name_pronoun_mismatch"
    return out
