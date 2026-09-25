"""Tests for timing_calculator word selection and mapping.

The reader builds ``current_sentence_words`` with one filter:

    [token for token in text.split() if re.search(r'[a-zA-Z0-9]', token)]

while timing math builds ``original_words`` with
``_get_highlightable_words``. Those two must agree **position by position**,
because ``word_mapping`` produces indices into the latter and the reader
subscripts the former with them. If the counts diverge, every word after the
divergent token resolves to an index one too high and the highlight lands on
the wrong word for the rest of the sentence.
"""

from __future__ import annotations

import re

import pytest

from lue import content_parser
from lue.timing_calculator import (
    _get_highlightable_words,
    _sanitize_word,
    create_word_mapping,
    mapping_for_displayed_words,
    owner_word_index,
    process_tts_timing_data,
    select_highlight_word,
)


def reader_words(text: str) -> list[str]:
    """Exact expression from reader.py's _new_sentence_started handler."""
    return [token for token in text.split() if re.search(r"[a-zA-Z0-9]", token)]


def spoken_words(sentence: str) -> list[str]:
    """The words the TTS engine is handed for this sentence."""
    return _get_highlightable_words(content_parser.sanitize_text_for_tts(sentence))


def timing_info_for(sentence: str) -> dict:
    """timing_info exactly as a provider builds it: from the *spoken* text."""
    raw = [(w, i * 0.25, i * 0.25 + 0.25) for i, w in enumerate(spoken_words(sentence))]
    return process_tts_timing_data(
        content_parser.sanitize_text_for_tts(sentence),
        raw,
        total_duration=len(raw) * 0.25,
    )


def highlight_sequence(sentence: str) -> list[int]:
    """Token index the reader highlights at the middle of every spoken word."""
    info = timing_info_for(sentence)
    displayed = reader_words(sentence)
    mapping = mapping_for_displayed_words(displayed, info)
    assert mapping is not None
    return [
        select_highlight_word(info["word_timings"], mapping, i * 0.25 + 0.125, len(displayed))
        for i in range(len(info["word_timings"]))
    ]


def expected_owner(sentence: str) -> list[int]:
    """Independent ground truth: which displayed token each spoken word is.

    A displayed token is exactly the concatenation of the spoken words it was
    split into, so walk both lists accumulating until they spell the token.
    """
    displayed, spoken = reader_words(sentence), spoken_words(sentence)
    owner: list[int] = []
    i = 0
    for k, token in enumerate(displayed):
        want = _sanitize_word(token)
        acc, start = "", i
        while i < len(spoken) and len(acc) < len(want):
            acc += _sanitize_word(spoken[i])
            i += 1
            if acc == want:
                break
        assert acc == want, f"{token!r}: {acc!r} != {want!r}"
        owner.extend([k] * (i - start))
    assert i == len(spoken), f"{len(spoken) - i} spoken words unclaimed"
    return owner


# Standalone non-ASCII punctuation: counted by the old filter, ignored by the
# reader and never emitted as a word by TTS engines.
DIVERGENT = [
    "PerfectBound ™ and the PerfectBound™ logo are trademarks.",
    "These are the “stuff ” out of which this book has come.",
    "Then it became popularized as “what if ” analysis",
    "Here are some of them: • First problem: the company has no expertise.",
    "Oh lost — come back again to the wind-grieved ghosts, my friends…",
]


class TestFilterAgreement:
    @pytest.mark.parametrize("text", DIVERGENT)
    def test_counts_match_reader(self, text):
        assert len(_get_highlightable_words(text)) == len(reader_words(text))

    def test_standalone_trademark_excluded(self):
        assert "™" not in _get_highlightable_words("PerfectBound ™ logo")

    def test_standalone_quotes_and_bullets_excluded(self):
        assert _get_highlightable_words("a ” b") == ["a", "b"]
        assert _get_highlightable_words("list • item") == ["list", "item"]

    def test_non_ascii_letters_still_excluded_like_reader(self):
        # reader's test is ASCII-only; the two must agree either way.
        text = "shalom שלום world"
        assert len(_get_highlightable_words(text)) == len(reader_words(text))

    @pytest.mark.parametrize("text", [
        "Crossing the Chasm was written in 1990 and published in 1991.",
        "“Oh lost and by the wind-grieved ghosts, come back again!”",
        "Before—what later became known as “what if” analysis",
        "$5 for 3 apples, about 7 PM!",
        "don't stop —believing— now",
    ])
    def test_plain_sentences_unchanged(self, text):
        """Ordinary prose must still produce one entry per word."""
        assert len(_get_highlightable_words(text)) == len(reader_words(text))


class TestCreateWordMapping:
    def test_perfect_match_returns_identity(self):
        words = ["the", "quick", "brown", "fox"]
        timings = [(w, i * 0.5, i * 0.5 + 0.5) for i, w in enumerate(words)]
        assert create_word_mapping(words, timings) == [0, 1, 2, 3]

    def test_empty_inputs_return_none(self):
        assert create_word_mapping([], []) is None
        assert create_word_mapping(["a"], []) is None

    def test_merged_words_map_to_shared_timing(self):
        """Multiple source words sharing one TTS chunk all point at it."""
        words = ["October", "2001"]
        timings = [("October 2001", 0.0, 1.5)]
        assert create_word_mapping(words, timings) == [0, 0]

    def test_no_cascade_after_merge(self):
        words = ["October", "2001", "was", "published", "in", "1991"]
        timings = [
            ("October 2001", 0.0, 1.5),
            ("was", 1.5, 1.8),
            ("published", 1.8, 2.5),
            ("in", 2.5, 2.7),
            ("1991", 2.7, 3.3),
        ]
        assert create_word_mapping(words, timings) == [0, 0, 1, 2, 3, 4]

    def test_mapping_indices_stay_within_timings_range(self):
        words = ["a", "b", "c", "d"]
        timings = [("a", 0, 1), ("b", 1, 2)]
        mapping = create_word_mapping(words, timings)
        assert mapping is not None
        assert all(0 <= i < len(timings) for i in mapping)


class TestSpokenVsDisplayedTokens:
    """``sanitize_text_for_tts`` rewrites text before speaking it.

    Every provider builds ``timing_info`` from that rewrite, so its
    ``word_mapping`` indexes the *spoken* tokens while the reader highlights
    the *displayed* ones. Rebuilding the mapping against the displayed tokens
    is what keeps the two in step.
    """

    # (sentence, expected mapping, expected highlight per spoken word)
    CASES = [
        (
            "I only read get-rich-quick schemes.",
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 3, 3, 4],
        ),
        (
            "We met in the past—it was sunny.",
            [0, 1, 2, 3, 4, 6, 7],
            [0, 1, 2, 3, 4, 4, 5, 6],
        ),
        (
            "See pages 10–20 for details.",
            [0, 1, 2, 4, 5],
            [0, 1, 2, 2, 3, 4],
        ),
        (
            "All rights reserved under International and Pan-American Copyright Conventions.",
            [0, 1, 2, 3, 4, 5, 6, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 6, 7, 8],
        ),
    ]

    @pytest.mark.parametrize("sentence,mapping_expected,sequence", CASES)
    def test_mapping_is_built_in_displayed_token_space(self, sentence, mapping_expected, sequence):
        displayed = reader_words(sentence)
        info = timing_info_for(sentence)
        mapping = mapping_for_displayed_words(displayed, info)
        assert mapping == mapping_expected
        # one entry per displayed token, each addressing a spoken word
        assert len(mapping) == len(displayed)
        assert max(mapping) < len(info["word_timings"])
        # ... and the selection is always a usable displayed index
        assert highlight_sequence(sentence) == sequence
        assert set(sequence) <= set(range(len(displayed)))

    @pytest.mark.parametrize("sentence", [
        "I only read get-rich-quick schemes.",
        "We met in the past—it was sunny.",
        "See pages 10–20 for details.",
        "The state-of-the-art mill cost $2 billion.",
        "All rights reserved under International and Pan-American Copyright Conventions.",
        "“state-of-the-art” when the pragmatist wants “industry standard.”",
        "Don't use a one-size-fits-all strategy—use a pay-as-you-go budget.",
        "Crossing the Chasm was written in 1990 and published in 1991.",
        "PerfectBound ™ and the PerfectBound™ logo are trademarks.",
        "Oh lost — come back again to the wind-grieved ghosts, my friends…",
    ])
    def test_highlight_tracks_the_spoken_word(self, sentence):
        """The reader must land on the token that contains the spoken word."""
        assert highlight_sequence(sentence) == expected_owner(sentence)

    def test_timing_info_mapping_indexes_the_spoken_tokens(self):
        """Regression guard: timing_info's own mapping cannot be used directly.

        It counts the words the engine spoke, and the engine was handed the
        sanitized text, where ``get-rich-quick`` is three words. The reader
        indexes displayed tokens with it, so its trailing entries address
        tokens that do not exist and the words before them land ahead of the
        one being spoken.
        """
        sentence = "I only read get-rich-quick schemes."
        info = timing_info_for(sentence)
        spoken, displayed = spoken_words(sentence), reader_words(sentence)
        assert len(spoken) == 7
        assert len(displayed) == 5
        assert info["word_mapping"] == list(range(7))
        assert max(info["word_mapping"]) >= len(displayed)

        # Taken at face value it highlights "schemes." while "get" is spoken
        # and runs off the end of the sentence afterwards.
        direct = [
            select_highlight_word(info["word_timings"], info["word_mapping"], i * 0.25 + 0.125, 5)
            for i in range(7)
        ]
        assert direct == [0, 1, 2, 3, 4, 4, 4]
        assert highlight_sequence(sentence) == [0, 1, 2, 3, 3, 3, 4]

    def test_untouched_sentences_keep_an_identity_mapping(self):
        sentence = "Crossing the Chasm was written in 1990 and published in 1991."
        assert content_parser.sanitize_text_for_tts(sentence) == sentence
        displayed = reader_words(sentence)
        assert mapping_for_displayed_words(displayed, timing_info_for(sentence)) == list(
            range(len(displayed))
        )

    def test_no_timings_yields_no_mapping(self):
        assert mapping_for_displayed_words(["a", "b"], {"word_timings": []}) is None
        assert mapping_for_displayed_words(["a", "b"], {}) is None


class TestOwnerWordIndex:
    def test_returns_the_token_containing_the_spoken_word(self):
        # token 3 spans spoken words 3, 4 and 5
        mapping = [0, 1, 2, 5]
        assert [owner_word_index(mapping, t) for t in range(6)] == [0, 1, 2, 2, 2, 3]

    def test_before_the_first_mapped_word(self):
        assert owner_word_index([2, 3], 0) == 0
        assert owner_word_index([2, 3], 1) == 0
        assert owner_word_index([2, 3], 2) == 0
        assert owner_word_index([2, 3], 3) == 1


class TestSelectHighlightWord:
    """The reader's ``_word_update_loop`` selection, extracted for testing."""

    def test_no_timings_estimates_evenly(self):
        assert select_highlight_word(None, None, 0.5, 4, sentence_duration=2.0) == 1
        assert select_highlight_word([], [], 99.0, 4, sentence_duration=2.0) == 3
        assert select_highlight_word(None, None, 0.5, 4, sentence_duration=0.0) == 0
        assert select_highlight_word(None, None, 0.5, 0, sentence_duration=2.0) == 0

    def test_without_mapping_uses_the_spoken_index(self):
        timings = [("a", 0.0, 1.0), ("b", 1.0, 2.0)]
        assert select_highlight_word(timings, None, 0.5, 2) == 0
        assert select_highlight_word(timings, None, 1.5, 2) == 1
        # past the end: stay on the last token
        assert select_highlight_word(timings, None, 9.0, 2) == 1
        # clamped when the timings outnumber the displayed tokens
        assert select_highlight_word(timings, None, 1.5, 1) == 0

    def test_sub_word_split_of_a_shared_timing(self):
        timings = [("October 2001", 0.0, 1.0)]
        assert select_highlight_word(timings, [0, 0], 0.2, 2) == 0
        assert select_highlight_word(timings, [0, 0], 0.8, 2) == 1
        # degenerate timing must not divide by zero
        assert select_highlight_word([("a b", 1.0, 1.0)], [0, 0], 1.0, 2) == 0

    def test_gap_in_the_mapping_stays_on_its_token(self):
        # token 3 starts at spoken word 5; spoken words 3 and 4 have no entry
        # of their own and belong to token 2
        timings = [(f"w{i}", float(i), float(i + 1)) for i in range(6)]
        mapping = [0, 1, 2, 5]
        assert select_highlight_word(timings, mapping, 2.5, 4) == 2
        assert select_highlight_word(timings, mapping, 3.5, 4) == 2
        assert select_highlight_word(timings, mapping, 4.5, 4) == 2
        assert select_highlight_word(timings, mapping, 5.5, 4) == 3

    def test_indices_are_clamped_to_the_displayed_sentence(self):
        # a mapping longer than the displayed sentence (stale/foreign data)
        timings = [("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0)]
        assert select_highlight_word(timings, [0, 1, 2], 0.5, 2) == 0
        assert select_highlight_word(timings, [0, 1, 2], 2.5, 2) == 1


class TestProcessTimingData:

    def test_anchored_output_maps_1_to_1(self):
        text = "October 2001 ISBN 0-06-018987-8 was published in 1991."
        words = _get_highlightable_words(text)
        timings = [(w, i * 0.4, i * 0.4 + 0.4) for i, w in enumerate(words)]

        info = process_tts_timing_data(text, timings, total_duration=3.0)

        assert info["word_mapping"] == list(range(len(words)))
        assert info["word_mapping"] == list(range(len(reader_words(text))))

    def test_length_agrees_with_reader_for_divergent_text(self):
        for text in DIVERGENT:
            info = process_tts_timing_data(text, [], total_duration=2.0)
            # Fallback estimation must cover exactly the reader's words.
            assert len(info["word_timings"]) == len(reader_words(text))
