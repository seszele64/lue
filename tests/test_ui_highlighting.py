"""Tests for the word-highlight unit iterator in ui.py.

``reader.ui_word_idx`` indexes ``reader.current_sentence_words`` (one entry per
whitespace token containing an ASCII alphanumeric). The renderer must walk the
same index space, or the highlight lands on the wrong token. An earlier
implementation split tokens on hyphens/em dashes, which made the counter run
wider than ``ui_word_idx`` on 20.4% of sentences.
"""

from __future__ import annotations

import re

import pytest

from lue.ui import iter_highlight_units


def reader_words(text: str) -> list[str]:
    """Exact expression from reader.py's _new_sentence_started handler."""
    return [token for token in text.split() if re.search(r"[a-zA-Z0-9]", token)]


def old_ui_count(text: str) -> int:
    """The previous, buggy sub-part counter -- kept to document the delta."""
    n = 0
    for token in text.lstrip().split():
        for part in re.split(r"([—-])", token):
            if part in ["—", "-"] or not re.search(r"[a-zA-Z0-9]", part):
                continue
            n += 1
    return n


HYPHENATED = [
    "By payment of the required fees, you have been granted the "
    "non-exclusive, non-transferable right to access and read this e-book "
    "on-screen.",
    "All rights reserved under International and Pan-American Copyright Conventions.",
    "Adobe Acrobat E-Book Reader edition v 1.",
    "A so-called long-term, high-tech get-rich-quick opportunity.",
    "Use the em—dash sparingly in well—written prose.",
]


class TestUnitCount:
    @pytest.mark.parametrize("text", HYPHENATED)
    @pytest.mark.parametrize("text2", ["a ” b", "list • item", "ends.” Then next"])
    def test_every_index_selects_the_readers_token(self, text, text2):
        """The real contract: ui_word_idx -> same token the reader addresses."""
        for sentence in (text, text2):
            words = reader_words(sentence)
            for target in range(len(words)):
                selected = [tok for tok, cur in
                            iter_highlight_units(sentence, target) if cur]
                assert selected == [words[target]], (
                    f"index {target} selected {selected}, reader has {words[target]}"
                )

    @pytest.mark.parametrize("text", HYPHENATED)
    def test_index_space_is_not_inflated_by_hyphens(self, text):
        """An out-of-range index must select nothing (no wraparound)."""
        words = reader_words(text)
        assert not any(cur for _t, cur in iter_highlight_units(text, len(words)))
        assert not any(cur for _t, cur in iter_highlight_units(text, 10_000))

    @pytest.mark.parametrize("text", HYPHENATED)
    def test_differs_from_buggy_sub_part_counter(self, text):
        """Guard: if this stops holding, the regression has returned."""
        assert old_ui_count(text) > len(reader_words(text))

    @pytest.mark.parametrize("text", HYPHENATED)
    def test_every_index_reachable_exactly_once(self, text):
        for target in range(len(reader_words(text))):
            units = list(iter_highlight_units(text, target))
            assert sum(1 for _tok, cur in units if cur) == 1


class TestTokenHandling:
    def test_hyphenated_token_is_one_unit(self):
        units = list(iter_highlight_units("a long-term b", 1))
        assert [t for t, _ in units] == ["a", "long-term", "b"]
        assert [t for t, cur in units if cur] == ["long-term"]

    def test_em_dash_token_is_one_unit(self):
        units = list(iter_highlight_units("before—what happened", 0))
        assert [t for t, _ in units] == ["before—what", "happened"]

    def test_punctuation_only_tokens_never_consume_an_index(self):
        units = list(iter_highlight_units("a ” b", 1))
        assert [t for t, _ in units] == ["a", "”", "b"]
        # The closing quote is not a word, so index 1 is 'b', not '”'.
        assert [t for t, cur in units if cur] == ["b"]

    def test_leading_whitespace_is_stripped_like_before(self):
        units = list(iter_highlight_units("   indented words here", 0))
        assert [t for t, _ in units] == ["indented", "words", "here"]

    @pytest.mark.parametrize("text,expected", [
        ("", []),
        # Still yielded so it renders, but is_current stays False (see below).
        ("...", ["..."]),
        ("one", ["one"]),
        ("$5 for 3 apples, about 7 PM!", ["$5", "for", "3", "apples,", "about", "7", "PM!"]),
    ])
    def test_basic_tokenisation(self, text, expected):
        assert [t for t, _ in iter_highlight_units(text, 0)] == expected

    @pytest.mark.parametrize("text", ["...", "”", "— • …"])
    def test_punctuation_tokens_never_highlight(self, text):
        units = list(iter_highlight_units(text, 0))
        assert all(not cur for _t, cur in units)
