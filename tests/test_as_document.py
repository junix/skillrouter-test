"""Contract tests for ``Skill.as_document`` — the ``name | description | body`` flattener.

The contract (from models.py):
  format    = "{name} | {description[:desc_max]} | {body[:body_max]}"
  desc_max  defaults to 500, body_max to 2000
  slicing   is Python str slicing (by Unicode code point, not byte), exclusive of the cap
  delimiter " | " is always emitted even when the adjacent field is empty
"""

import pytest

from skillrouter_test.models import Skill


# --- format / default caps -------------------------------------------------

def test_full_document_uses_defaults_and_joins_with_pipe_spaces() -> None:
    doc = Skill("speech-to-text", "Transcribe audio.", "Runs whisper.").as_document()
    assert doc == "speech-to-text | Transcribe audio. | Runs whisper."


def test_empty_description_and_body_still_keep_delimiters() -> None:
    # The separators are structural; empty fields collapse to a leading/trailing space,
    # never to "name|" or "name| |".
    assert Skill("only-name").as_document() == "only-name |  | "


def test_pipes_in_field_values_are_not_treated_as_delimiters() -> None:
    # Pipe chars inside data must pass through verbatim (no splitting/escaping).
    doc = Skill("a|b", "c|d", "e|f").as_document()
    assert doc == "a|b | c|d | e|f"


# --- truncation boundaries -------------------------------------------------

@pytest.mark.parametrize(
    ("desc_max", "expected_desc"),
    [
        (0, ""),          # zero keeps nothing
        (1, "a"),         # single char
        (3, "abc"),       # exact length — boundary, exclusive slice
        (4, "abc"),       # length 3, cap 4 — no padding, no ellipsis
        (10, "abc"),      # cap well above length — unchanged
    ],
)
def test_description_truncation_boundaries(desc_max: int, expected_desc: str) -> None:
    doc = Skill("n", "abc", "x").as_document(desc_max=desc_max, body_max=10)
    assert doc == f"n | {expected_desc} | x"


@pytest.mark.parametrize(
    ("body_max", "expected_body"),
    [(0, ""), (1, "x"), (3, "xyz"), (4, "xyz"), (100, "xyz")],
)
def test_body_truncation_boundaries(body_max: int, expected_body: str) -> None:
    doc = Skill("n", "abc", "xyz").as_document(desc_max=100, body_max=body_max)
    assert doc == f"n | abc | {expected_body}"


def test_truncation_cuts_at_exact_cap_not_cap_minus_one() -> None:
    # Off-by-one guard: cap == len(slice) keeps all chars, does NOT drop the last.
    doc = Skill("n", "12345", "12345").as_document(desc_max=5, body_max=5)
    assert doc == "n | 12345 | 12345"


def test_truncation_cuts_at_cap_plus_one_drops_one_char() -> None:
    doc = Skill("n", "12345", "12345").as_document(desc_max=4, body_max=4)
    assert doc == "n | 1234 | 1234"


# --- Unicode / slicing semantics ------------------------------------------

def test_unicode_is_sliced_by_code_point_not_byte() -> None:
    # "é" is one code point; a byte-based slice at cap 2 on "ééé" would mangle UTF-8.
    doc = Skill("n", "ééé", "ññ").as_document(desc_max=2, body_max=1)
    assert doc == "n | éé | ñ"


def test_multibyte_emoji_counts_as_one_code_point() -> None:
    # Each emoji is a single code point; cap 2 keeps two glyphs.
    doc = Skill("n", "😀😁😂", "").as_document(desc_max=2, body_max=10)
    assert doc == "n | 😀😁 | "


# --- independent cap application ------------------------------------------

def test_desc_and_body_caps_are_independent() -> None:
    doc = Skill("n", "abcdefgh", "xyz").as_document(desc_max=3, body_max=100)
    assert doc == "n | abc | xyz"


def test_name_is_never_truncated() -> None:
    # The contract slices only description and body; name passes through whole.
    long_name = "x" * 100
    assert Skill(long_name, "", "").as_document(desc_max=0, body_max=0) == f"{long_name} |  | "
