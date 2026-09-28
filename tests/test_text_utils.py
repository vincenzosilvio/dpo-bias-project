"""Filters and name detection. Regressions from past review rounds are marked."""
from text_utils import analyze, exclusion_reason, find_names, get_nlp, has_first_person_narration


def rec(text, **kw):
    return {"completion": text, "truncated": False, **kw}


def test_mixed_pronouns_excluded():
    # lesson #7: "5 she + 4 he" used to count as counter-stereotypical
    assert exclusion_reason(rec("She fixed it. " * 5 + "He watched. " * 4), 2) == "mixed"


def test_text_exclusions():
    assert exclusion_reason(rec("As an AI language model, she ..."), 1) == "refusal"
    assert exclusion_reason(rec("Mark Zuckerberg walked in. He smiled."), 1) == "real_person"
    assert exclusion_reason({"completion": "She ran. She sat.", "truncated": True}, 1) == "truncated"
    assert exclusion_reason(rec("She ran."), 2) == "no_signal"
    assert exclusion_reason(rec("She ran. She sat."), 2) is None


def test_first_person_only_outside_dialogue():
    assert not has_first_person_narration('She said, "I love my job." She smiled.')
    assert has_first_person_narration("I watched her work. She was fast.")
    assert has_first_person_narration("My colleague said she was fast.")


def test_surname_is_not_a_second_person():
    # "Chen" and "Hartley" are also first names in the corpus
    nlp = get_nlp()
    assert find_names(nlp("Nurse Emily Chen checked her charts. She left.")) == {"Emily": "female"}
    assert find_names(nlp("Dr. Amelia Hartley reviewed her plans. She smiled.")) == {"Amelia": "female"}


def test_analyze_single_character():
    ok = analyze(rec("Marcus locked the garage. He drove home, tired from his shift."), 2)
    assert ok["reason"] is None and ok["gender"] == "male" and ok["name"] == "Marcus"


def test_analyze_rejects_second_person():
    a = analyze(rec("Timmy met Sarah at the shop. He waved and he left."), 2)
    assert a["reason"] == "multi_person"


def test_analyze_rejects_name_pronoun_mismatch():
    # lesson #9: pronouns belonging to someone other than the named holder
    a = analyze(rec("Sarah finished the report. He smiled at his desk."), 2)
    assert a["reason"] == "name_pronoun_mismatch"


def test_unisex_name_is_usable():
    a = analyze(rec("Alex tightened the bolt. She wiped her hands."), 2)
    assert a["reason"] is None and a["name_gender"] == "unisex"


def test_sentence_initial_ly_name_detected():
    # spaCy tags "Emily" at sentence start as ADV (found via review section 5)
    nlp = get_nlp()
    assert find_names(nlp("Emily finished the last task. She packed her tools.")) == {"Emily": "female"}


def test_everyday_word_not_a_name():
    nlp = get_nlp()
    assert find_names(nlp("Hope filled the room as Marcus opened his laptop.")) == {"Marcus": "male"}


def test_quoted_slogan_is_not_a_name():
    # lesson #10: "Always Happy" on a hat was taken as the CEO's name
    nlp = get_nlp()
    t = 'The CEO wore her favorite red hat that said "Always Happy." She smiled.'
    assert find_names(nlp(t)) == {}


def test_titled_surname_gets_title_gender():
    nlp = get_nlp()
    assert find_names(nlp("Mr. Johnson, the CEO, reviewed his plans. He left.")) == {"Johnson": "male"}


def test_single_pronoun_usable_when_name_agrees():
    a = analyze(rec("CEO Emily Chen solved her biggest crisis at work."), 1)
    assert a["reason"] is None and a["gender"] == "female"


def test_quoted_nickname_does_not_split_name():
    # smoke test #2: 'Elisabeth "Betty" Rogers' was counted as two people
    nlp = get_nlp()
    t = 'Elisabeth "Betty" Rogers was a busy executive. She woke up early.'
    assert find_names(nlp(t)) == {"Elisabeth": "female"}


# ---- regressions from the first full short-form review (2026-09-28) ------

def test_titled_surname_uses_title_gender():
    # "Mrs. Smith ... her" was excluded as name_pronoun_mismatch ("Smith" is
    # a male first name in the corpus).
    from text_utils import find_names, get_nlp
    nlp = get_nlp()
    assert find_names(nlp("Mrs. Smith arrives early in her raincoat.")) == {"Smith": "female"}
    assert find_names(nlp("Ms. Patel woke up early on her first day.")) == {"Patel": "female"}


def test_blocklist_catches_michael_jackson():
    from text_utils import mentions_real_person
    assert mentions_real_person("Michael Jackson's life ended that evening.")


def test_blocklist_matches_whole_words_only():
    from text_utils import mentions_real_person
    assert not mentions_real_person("Mechanic Jack made his way through the crowd.")
    assert mentions_real_person("Jack Ma founded a company.")
