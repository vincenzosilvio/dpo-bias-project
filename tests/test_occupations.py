"""Prompt-level regressions."""
import re

from occupations import build_prompts

PRONOUNS = re.compile(r"\b(he|she|him|her|his|hers)\b", re.IGNORECASE)


def test_short_prompts_never_name_pronouns():
    # lesson #10: with "he or she" / "she or he" in the prompt, the model used
    # whichever pronoun came first -- measuring word order, not occupation.
    for p in build_prompts("short"):
        assert not PRONOUNS.search(p["prompt"]), p["prompt"]


def test_short_prompts_ask_for_name_first():
    for p in build_prompts("short"):
        assert "Begin the text with" in p["prompt"]


def test_template_splits():
    ps = build_prompts("short")
    assert {p["template_split"] for p in ps} == {"train", "heldout"}
    assert {p["template_split"] for p in build_prompts("longform")} == {"eval"}
