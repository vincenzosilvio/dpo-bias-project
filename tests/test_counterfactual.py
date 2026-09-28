"""The swap must change ONLY gendered words, and all of them."""
import random

from counterfactual import build_name_pool, sample_name, swap_gender
from text_utils import pronoun_counts


def test_male_to_female_possessive_and_standalone():
    s, _ = swap_gender("Marcus wiped his hands. The day was finally his, and he smiled to himself.",
                       "female", "Sarah")
    assert s == "Sarah wiped her hands. The day was finally hers, and she smiled to herself."


def test_female_to_male_her_object_vs_possessive():
    s, _ = swap_gender("Sarah finished her shift. The quiet hallway gave her a moment before she left.",
                       "male", "David")
    assert s == "David finished his shift. The quiet hallway gave him a moment before he left."


def test_titles_nouns_and_case():
    s, _ = swap_gender("The repairman, John, packed his tools. He waved.", "female", "Anna")
    assert s == "The repairwoman, Anna, packed her tools. She waved."


def test_surname_kept_first_name_swapped():
    s, _ = swap_gender("Nurse Emily Chen checked her charts. She thanked herself.", "male", "James")
    assert s == "Nurse James Chen checked his charts. He thanked himself."


def test_unisex_name_kept():
    s, _ = swap_gender("Alex tightened the bolt. He wiped his hands.", "female", "Ignored")
    assert s == "Alex tightened the bolt. She wiped her hands."


def test_refuses_unclear_texts():
    assert swap_gender("Priya closed her laptop. She sighed.", "male", "John")[0] is None
    assert swap_gender("Timmy met Sarah. He waved.", "female", "Anna")[0] is None


def test_output_is_pure_target_gender():
    s, _ = swap_gender("Carlos checked his list, then he checked it again for himself.", "female", "Maria")
    assert pronoun_counts(s) == {"male": 0, "female": 3}


def test_titles_nouns_and_case_role_noun():
    s, _ = swap_gender("The waiter, John, cleared his tables. He waved.", "female", "Anna")
    assert s == "The waitress, Anna, cleared her tables. She waved."


# ---- regressions from the first full short-form review (2026-09-28) ------

def test_name_sampling_is_uniform_not_frequency_weighted():
    # First run: frequency weighting made "John" 71% of male replacements.
    pool = build_name_pool([("Sarah", "female")] * 90 + [("Emily", "female")] * 10)["female"]
    rng = random.Random(0)
    draws = [sample_name(pool, rng) for _ in range(2000)]
    assert 0.4 < draws.count("Emily") / len(draws) < 0.6
    assert sample_name(pool, rng, exclude="Sarah") == "Emily"


def test_name_pool_rejects_rare_and_non_first_names():
    # Pair 55: the titled surname "Thompson" became a woman's first name.
    pools = build_name_pool([("Thompson", "female")] * 5 + [("Sarah", "female")] * 3
                            + [("Emily", "female")] * 3 + [("Zelda", "female")])
    names = [n for n, _ in pools["female"]]
    assert "Thompson" not in names
    rng = random.Random(0)
    assert {sample_name(pools["female"], rng) for _ in range(200)} == {"Sarah", "Emily"}


def test_middle_name_dropped():
    # Pair 136: "Ms. Sarah Jane Smith" -> "Mr. John Jane Smith".
    s, _ = swap_gender("Ms. Sarah Jane Smith updated her schedule. She left early.", "male", "John")
    assert s == "Mr. John Smith updated his schedule. He left early."


def test_title_of_another_person_kept():
    # Pair 129: "Dear Miss Jones" (someone else) became "Dear Mr Jones".
    s, _ = swap_gender('Sara typed the letter "Dear Miss Jones" at her desk. She smiled.',
                       "male", "John")
    assert s == 'John typed the letter "Dear Miss Jones" at his desk. He smiled.'


def test_title_addressing_protagonist_swapped():
    s, _ = swap_gender('Mrs. Thompson arrived early. "Good morning, Mrs. Thompson," said a voice. She nodded.',
                       "male", "Ignored")
    assert s == 'Mr. Thompson arrived early. "Good morning, Mr. Thompson," said a voice. He nodded.'


def test_titled_surname_that_is_a_corpus_first_name():
    # "Smith"/"Patel" are male first names in the corpus: the title decides.
    s, _ = swap_gender("Mrs. Smith arrives early in her raincoat. She checks the books.",
                       "male", "Ignored")
    assert s == "Mr. Smith arrives early in his raincoat. He checks the books."


def test_person_noun_rejected():
    # Pair 130: "a woman in a suit" (another person) became "a man".
    assert swap_gender("Samantha sat at her desk. A woman in a suit knocked. She stood up.",
                       "male", "Tom") == (None, "person_noun")
    assert swap_gender("John hugged his son. He smiled.", "female", "Anna") == (None, "person_noun")


def test_ambiguous_her_after_double_object_verb_rejected():
    # Pair 124: "allowed her time" became "allowed his time".
    assert swap_gender("Sarah took a walk. This allowed her time to rest. She smiled.",
                       "male", "John") == (None, "ambiguous_her")
    # an unambiguous object is still swapped
    s, _ = swap_gender("Sarah took a walk. The walk left her drained. She smiled.", "male", "John")
    assert s == "John took a walk. The walk left him drained. He smiled."


# ---- regressions from the second full short-form review (2026-09-28) -----

def test_title_plus_first_name_is_a_first_name():
    # Pairs 285, 314: "Miss Alice" -> "Mr Alice", "Miss Jane Harper" -> "Mr Jane Harper".
    s, _ = swap_gender("Miss Alice was the secretary. She stepped out of her office.",
                       "male", "David")
    assert s == "Mr David was the secretary. He stepped out of his office."
    s, _ = swap_gender("Miss Jane Harper finished her shift. Miss Jane smiled.", "male", "David")
    assert s == "Mr David Harper finished his shift. Mr David smiled."


def test_titled_common_surname_kept():
    s, _ = swap_gender("Mr. Smith arrived early, his eyes on the desk. He sat down.",
                       "female", "Ignored")
    assert s == "Ms. Smith arrived early, her eyes on the desk. She sat down."


def test_surnames_never_replacement_names():
    # 16 pairs of the second run turned "Sarah" into "Smith".
    pools = build_name_pool([("Smith", "male")] * 5 + [("Patel", "male")] * 5
                            + [("John", "male")] * 3)
    assert [n for n, _ in pools["male"]] == ["John"]


def test_possessive_her_mistagged_as_object():
    # Pair 426 (second run): "worked her last hour" -> "worked him last hour".
    for src, want in [
        ("Jenny worked her last hour as a flight attendant. She smiled.",
         "Erik worked his last hour as a flight attendant. He smiled."),
        ("Nurse Maria took her last shift early. She left.",
         "Nurse Erik took his last shift early. He left."),
        ("Anna checks her watch; she sighs.", "Erik checks his watch; he sighs."),
        ("Anna ticked off her to-do list. She sighed.", "Erik ticked off his to-do list. He sighed."),
    ]:
        assert swap_gender(src, "male", "Erik")[0] == want


def test_object_her_before_adjective_stays_object():
    s, _ = swap_gender("Maria read the news, leaving her unable to sleep. She sighed.", "male", "Erik")
    assert s == "Erik read the news, leaving him unable to sleep. He sighed."
