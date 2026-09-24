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


def test_name_sampling_follows_model_distribution():
    pool = build_name_pool([("Sarah", "female")] * 9 + [("Emily", "female")])["female"]
    rng = random.Random(0)
    draws = [sample_name(pool, rng) for _ in range(200)]
    assert draws.count("Sarah") > draws.count("Emily")
    assert sample_name(pool, rng, exclude="Sarah") == "Emily"


def test_titled_surname_swaps_title_keeps_surname():
    s, _ = swap_gender("Mr. Johnson, the CEO, reviewed his plans. He left early.", "female", "Ignored")
    assert s == "Ms. Johnson, the CEO, reviewed her plans. She left early."
