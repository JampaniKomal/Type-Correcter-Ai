"""Tests for the text corrector. NumPy only - no TensorFlow."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.autocorrect.predictor import Corrector  # noqa: E402


@pytest.fixture(scope="module")
def corrector():
    c = Corrector()
    assert c.loaded, "the model artifacts in model/ and data/ must load"
    return c


@pytest.mark.parametrize("bad,good", [
    ("teh", "the"),
    ("recieve", "receive"),
    ("beleive", "believe"),
    ("problme", "problem"),
    ("langauge", "language"),
])
def test_corrects_a_single_word(corrector, bad, good):
    assert corrector.correct_word(bad) == good


def test_corrects_a_whole_sentence(corrector):
    out = corrector.predict("teh recieve is beleive and i havw a problme")
    assert out == "the receive is believe and i have a problem"


def test_leaves_correct_text_unchanged(corrector):
    text = "this is a correct sentence that needs no changes"
    assert corrector.predict(text) == text


def test_does_not_touch_single_letter_words(corrector):
    # "a" and "i" are valid words the dictionary omits; they must survive.
    assert corrector.predict("i have a cat") == "i have a cat"


def test_preserves_capitalization(corrector):
    assert corrector.predict("Teh cat") == "The cat"
    assert corrector.predict("RECIEVE this") == "RECEIVE this"


def test_keeps_punctuation_and_spacing(corrector):
    out = corrector.predict("Hello, teh wrld!")
    assert out == "Hello, the world!"


def test_only_outputs_real_words(corrector):
    # Every output word is either the original token or a real dictionary word -
    # the corrector never emits an invented non-word.
    source = "zzzqx recieve somethign"
    out = corrector.predict(source)
    for original, corrected in zip(source.split(), out.split(), strict=True):
        assert corrected == original or corrected.lower() in corrector.dictionary
    assert "receive" in out and "something" in out
