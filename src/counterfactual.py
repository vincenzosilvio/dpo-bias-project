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
  * titles         Mr <-> Ms (Mrs/Miss -> Mr)
  * gendered nouns man/woman, -man/-woman compounds, waiter/waitress...

Only safe on texts that passed text_utils.analyze(): one named character,
pure pronouns. `swap_gender` re-checks its own output and returns None when
the result is not a clean opposite-gender text.
"""

import re

from text_utils import (
    MALE_PRONOUNS, FEMALE_PRONOUNS, find_names, get_nlp, name_gender, pronoun_counts,
)

PAIRS = [
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"),
    ("gentleman", "lady"), ("gentlemen", "ladies"), ("guy", "gal"),
    ("husband", "wife"), ("father", "mother"), ("dad", "mom"),
    ("son", "daughter"), ("brother", "sister"), ("uncle", "aunt"),
    ("nephew", "niece"), ("king", "queen"), ("sir", "ma'am"),
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


def _swap_token(tok, target, new_name, old_name):
    """Replacement text for one spaCy token, or None to keep it."""
    low = tok.lower_
    text = tok.text

    if old_name is not None and text == old_name:
        return new_name

    bare = text.rstrip(".")
    if bare in TITLE_TO_FEMALE and target == "female":
        return TITLE_TO_FEMALE[bare] + text[len(bare):]
    if bare in TITLE_TO_MALE and target == "male":
        return TITLE_TO_MALE[bare] + text[len(bare):]

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
    if g == "unisex":
        new_name, old_name = old_name, None   # a unisex name needs no swap
    elif name_gender(old_name) is None:
        # Titled surname ("Mr. Johnson"): gender comes from the title, which
        # is swapped below; the surname stays.
        new_name, old_name = old_name, None

    out, log = [], []
    for tok in doc:
        rep = _swap_token(tok, target, new_name, old_name)
        if rep is not None and rep != tok.text:
            log.append((tok.text, rep))
            out.append(rep + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    swapped = "".join(out)

    # Self-check: the result must be a clean text of the target gender.
    before, after = pronoun_counts(text), pronoun_counts(swapped)
    other = "male" if target == "female" else "female"
    if after[other] != 0 or after[target] != before[other] + before[target]:
        return None, "pronoun_check_failed"
    remaining = find_names(nlp(swapped))
    if old_name is not None and old_name in remaining:
        return None, "name_not_replaced"
    if any(gg not in (target, "unisex") for gg in remaining.values()):
        return None, "name_check_failed"
    return swapped, log


def build_name_pool(names_with_gender):
    """
    {gender: [(name, count), ...]} from the names the model produced, for
    sampling replacement names in-distribution.
    """
    from collections import Counter
    pools = {"male": Counter(), "female": Counter()}
    for name, g in names_with_gender:
        if g in pools:
            pools[g][name] += 1
    return {g: sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) for g, c in pools.items()}


def sample_name(pool, rng, exclude=None):
    names = [n for n, _ in pool if n != exclude]
    weights = [c for n, c in pool if n != exclude]
    return rng.choices(names, weights=weights, k=1)[0]


__all__ = ["swap_gender", "build_name_pool", "sample_name",
           "MALE_PRONOUNS", "FEMALE_PRONOUNS", "name_gender"]
