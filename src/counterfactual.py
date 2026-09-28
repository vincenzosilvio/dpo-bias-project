"""
Counterfactual gender swap (counterfactual data augmentation, cf. Lu et al.
2020; Zmigrod et al. 2019) for SHORT, SINGLE-CHARACTER completions.

Given a completion whose only character is gendered one way, produce the
same text with that character gendered the other way. The DPO pair is then
(rejected = original, chosen = swap): the two differ ONLY in gendered
words, so the preference signal is about gender and nothing else.

Swapped:
  * pronouns       he/she, him/his -> her, her -> his/him (by POS),
                   his (standalone) -> hers, himself/herself
  * the name       replaced by a name of the target gender, drawn from the
                   names the model itself produced (so the chosen text stays
                   in-distribution; odd corpus names would be a giveaway
                   feature DPO could learn instead of gender)
  * middle names   dropped ("Ms. Sarah Jane Smith" -> "Mr. John Smith")
  * titles         Mr <-> Ms (Mrs/Miss -> Mr), ONLY when they precede the
                   protagonist's name or surname ("Dear Miss Jones" is
                   someone else and stays)
  * role nouns     repairman/repairwoman, waiter/waitress... (they describe
                   the occupation-holder)

NOT swapped -- the text is rejected instead ("person_noun"): generic and
kinship nouns (man/woman, son/daughter, sir/ma'am...). They usually denote
ANOTHER person ("a woman in a suit", "his son"), where swapping changes a
second character; when they denote the protagonist, not swapping is wrong.
Either guess is wrong often, so these texts give no pair.

Also rejected ("ambiguous_her"): "her" right after a double-object verb
("allowed her time"): object or possessive, the parser can't tell.

Only safe on texts that passed text_utils.analyze(): one named character,
pure pronouns. `swap_gender` re-checks its own output and returns None when
the result is not a clean opposite-gender text.
"""

import re

from text_utils import (
    MALE_PRONOUNS, FEMALE_PRONOUNS, TITLE_GENDER, TITLES, find_names, get_nlp,
    name_gender, name_spans, pronoun_counts, titled_surname,
)

# Generic / kinship person nouns: the text is rejected (see docstring).
PERSON_NOUNS = {
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "gentleman", "gentlemen", "lady", "ladies", "guy", "guys", "gal", "gals",
    "husband", "wife", "father", "mother", "dad", "mom", "mum", "son", "sons",
    "daughter", "daughters", "brother", "brothers", "sister", "sisters",
    "uncle", "aunt", "nephew", "niece", "king", "queen", "sir", "ma'am", "madam",
    "grandfather", "grandmother", "grandpa", "grandma", "boyfriend", "girlfriend",
}

# Double-object verbs: "her" right after them is object OR possessive.
DOUBLE_OBJECT_VERBS = {
    # Only verbs where "V her N" is usually dative ("allowed her time").
    # Verbs like leave/make/find are mostly possessive ("leave her
    # classroom") and would reject good texts (pair 149 of the first run).
    "allow", "give", "offer", "bring", "lend", "send", "show", "tell", "grant",
    "buy", "cost", "owe", "hand", "teach", "promise", "wish", "deny", "spare",
    "afford", "award", "feed", "earn",
}

# Role nouns: swapped (they describe the occupation-holder).
PAIRS = [
    ("chairman", "chairwoman"), ("businessman", "businesswoman"),
    ("repairman", "repairwoman"), ("salesman", "saleswoman"),
    ("spokesman", "spokeswoman"), ("craftsman", "craftswoman"),
    ("handyman", "handywoman"), ("fireman", "firewoman"),
    ("policeman", "policewoman"), ("workman", "workwoman"),
    ("waiter", "waitress"), ("actor", "actress"), ("steward", "stewardess"),
    ("host", "hostess"), ("headmaster", "headmistress"),
]
TO_FEMALE = {m: f for m, f in PAIRS}
TO_MALE = {f: m for m, f in PAIRS}

TITLE_TO_FEMALE = {"Mr": "Ms"}
TITLE_TO_MALE = {"Mrs": "Mr", "Ms": "Mr", "Miss": "Mr"}

NOUN_TAGS_AFTER_POSSESSIVE = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS",
                              "CD", "VBG", "VBN", "RB"}


def match_case(src, dst):
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def _next_word(tok):
    """Next token that is not a lone '.' (spaCy may split "Mrs" "." )."""
    j = tok.i + 1
    while j < len(tok.doc) and tok.doc[j].text == ".":
        j += 1
    return tok.doc[j] if j < len(tok.doc) else None


def _swap_token(tok, target, new_name, old_name, identifiers):
    """Replacement text for one spaCy token, or None to keep it."""
    low = tok.lower_
    text = tok.text

    if old_name is not None and text == old_name:
        return new_name

    bare = text.rstrip(".")
    if bare in TITLE_GENDER:
        nxt = _next_word(tok)
        if nxt is None or nxt.text not in identifiers:
            return None        # a title of someone else ("Dear Miss Jones")
        if bare in TITLE_TO_FEMALE and target == "female":
            return TITLE_TO_FEMALE[bare] + text[len(bare):]
        if bare in TITLE_TO_MALE and target == "male":
            return TITLE_TO_MALE[bare] + text[len(bare):]
        return None

    if target == "female":
        if low == "he":
            return match_case(text, "she")
        if low == "him":
            return match_case(text, "her")
        if low == "himself":
            return match_case(text, "herself")
        if low == "his":
            # determiner ("his tools") -> her; standalone ("was his") -> hers
            nxt = tok.nbor(1) if tok.i + 1 < len(tok.doc) else None
            if tok.dep_ == "poss" or (nxt is not None and nxt.tag_ in NOUN_TAGS_AFTER_POSSESSIVE):
                return match_case(text, "her")
            return match_case(text, "hers")
        if low in TO_FEMALE:
            return match_case(text, TO_FEMALE[low])
    else:
        if low == "she":
            return match_case(text, "he")
        if low == "hers":
            return match_case(text, "his")
        if low == "herself":
            return match_case(text, "himself")
        if low == "her":
            # possessive determiner ("her tools") -> his; object ("thanked
            # her") -> him. POS-based; residual errors are measured by the
            # manual audit ("swap_correct" column).
            return match_case(text, "his" if tok.tag_ == "PRP$" else "him")
        if low in TO_MALE:
            return match_case(text, TO_MALE[low])
    return None


def swap_gender(text, target, new_name, nlp=None):
    """
    Rewrite `text` so its single character has gender `target`
    ('male'/'female'). Returns (swapped_text, log) or (None, reason) if
    the swap can't be done cleanly.
    """
    nlp = nlp or get_nlp()
    doc = nlp(text)
    names = find_names(doc)
    if len(names) != 1:
        return None, "not_single_name"
    (old_name, g), = names.items()
    if g is None:
        return None, "unknown_name"
    other = "male" if target == "female" else "female"

    if any(t.lower_ in PERSON_NOUNS for t in doc):
        return None, "person_noun"
    if target == "male" and any(
            t.lower_ == "her" and t.tag_ == "PRP$" and t.i > 0
            and doc[t.i - 1].lemma_.lower() in DOUBLE_OBJECT_VERBS for t in doc):
        return None, "ambiguous_her"

    # Protagonist identifiers: the first name, and the surname(s) of the
    # spans that contain it. Titles are swapped only in front of these.
    identifiers = {old_name}
    middle = set()   # token indices of middle names, dropped
    for toks in name_spans(doc, old_name):
        k = next(i for i, t in enumerate(toks) if t.text == old_name)
        if len(toks) > k + 1:
            identifiers.add(toks[-1].text)                       # surname
            middle |= {t.i for t in toks[k + 1:-1] if name_gender(t.text) in (g, other)}

    if g == "unisex" or titled_surname(doc, old_name) or name_gender(old_name) is None:
        # Unisex name: no swap needed. Titled surname ("Mrs. Smith"): the
        # title carries the gender and is swapped below; the surname stays.
        new_name, old_name = old_name, None

    out, log = [], []
    for tok in doc:
        if tok.i in middle:
            log.append((tok.text, ""))
            continue            # drop the middle name and its whitespace
        rep = _swap_token(tok, target, new_name, old_name, identifiers)
        if rep is not None and rep != tok.text:
            log.append((tok.text, rep))
            out.append(rep + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    swapped = "".join(out)

    # Self-check: the result must be a clean text of the target gender.
    before, after = pronoun_counts(text), pronoun_counts(swapped)
    if after[other] != 0 or after[target] != before[other] + before[target]:
        return None, "pronoun_check_failed"
    sdoc = nlp(swapped)
    remaining = find_names(sdoc)
    if old_name is not None and old_name in remaining:
        return None, "name_not_replaced"
    if any(gg not in (target, "unisex") for gg in remaining.values()):
        return None, "name_check_failed"
    for t in sdoc:   # no title of the old gender left before the protagonist
        if TITLE_GENDER.get(t.text.rstrip(".")) == other:
            nxt = _next_word(t)
            if nxt is not None and nxt.text in identifiers | {new_name}:
                return None, "title_check_failed"
    return swapped, log


MIN_NAME_COUNT = 2   # a name must be produced by the model at least twice


def build_name_pool(names_with_gender):
    """
    {gender: [(name, count), ...]} from the names the model produced, for
    sampling replacement names in-distribution. Only corpus first names of
    that gender: a titled surname ("Mrs. Thompson" -> "Thompson") is not a
    first name (pair 55 of the first run got "Thompson" as a woman's name).
    """
    from collections import Counter
    pools = {"male": Counter(), "female": Counter()}
    for name, g in names_with_gender:
        if g in pools and name_gender(name) == g:
            pools[g][name] += 1
    return {g: sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) for g, c in pools.items()}


def sample_name(pool, rng, exclude=None):
    """
    UNIFORM over the model's names seen >= MIN_NAME_COUNT times. Weighting
    by frequency made "John" 71% of male replacements in the first run, so
    DPO could learn "prefer the token John" instead of the gender contrast.
    """
    names = [n for n, c in pool if n != exclude and c >= MIN_NAME_COUNT]
    if not names:
        names = [n for n, _ in pool if n != exclude]
    return rng.choice(names)


__all__ = ["swap_gender", "build_name_pool", "sample_name",
           "MALE_PRONOUNS", "FEMALE_PRONOUNS", "name_gender"]
