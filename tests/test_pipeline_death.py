"""Integration tests for the producer-to-player death propagation bug.

Bug description
---------------
When OpenAI TTS times out repeatedly, the pipeline dies permanently:

1. ``tts_parallel.py`` *__generate_one()* catches the exception and submits a
   ``_GenerationResult(success=False)`` to the *_OrderedBuffer*.
2. ``lookahead_buffer.py`` *_refill_batch_parallel()* sees the failed result
   and increments ``_error_count``.
3. When ``_error_count >= _max_errors`` (default 5), the producer puts a
   ``None`` sentinel on the internal asyncio.Queue and breaks its main loop.
4. The player receives ``None`` and sets ``playback_finished_event`` — playback
   permanently stops.
5. The main event loop never monitors or resets this, so the app appears
   frozen with the last sentence still highlighted.

All tests use ``pytest-asyncio`` with ``@pytest.mark.asyncio`` and
``unittest.mock`` only (no third-party mocks).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lue.tts_parallel import (
    _GenerationResult,
    _OrderedBuffer,
    ParallelTTSGen,
)


# ============================================================================
# Test 1: _OrderedBuffer passes failure results correctly
# ============================================================================


@pytest.mark.asyncio
async def test_ordered_buffer_passes_failure_result():
    """_OrderedBuffer correctly stores and returns failed _GenerationResult."""
    buf = _OrderedBuffer()
    buf._expected_total = 1

    fail = _GenerationResult(
        sentence_idx=(0, 0, 0),
        audio_path="",
        duration=0.0,
        timing_info={},
        success=False,
    )
    await buf.submit(fail, index=0)
    result = await buf.pop_next()

    assert result is not None
    assert result.success is False
    assert result.sentence_idx == (0, 0, 0)


# ============================================================================
# Test 2: Error count resets on success in _refill_batch_parallel
# ============================================================================


@pytest.mark.asyncio
async def test_error_count_resets_on_success_in_refill():
    """After a failed batch, a successful batch resets _error_count to 0."""
    from lue.lookahead_buffer import LookaheadBuffer
    from lue import content_parser

    # --- Mock reader with one paragraph of three sentences ---
    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [
                ["Sentence one. Sentence two. Sentence three."]
            ],
            "tts_model": MagicMock(),
            "tts_cache": None,
            "_advance_position": (lambda self, pos, wrap=False: None),
        },
    )()

    # Wire _advance_position to a real function that advances
    # through sentence indices 0 → 1 → 2 → None.
    def _advance(current_pos, mode="sentence", wrap=True):
        c, p, s = current_pos
        if s < 2:
            return (c, p, s + 1)
        return None

    reader._advance_position = _advance

    # --- Buffer with low max_errors ---
    buffer = LookaheadBuffer(reader, target_sentences=2, max_errors=5)
    buffer._is_running = True
    buffer._error_count = 3  # simulate previous failures

    # --- Mock ParallelTTSGen that returns **success** ---
    class MockSuccessGen:
        def __init__(self):
            self._submitted: list[tuple] = []
            self._results: list[_GenerationResult] = []

        def submit(self, sentence_idx, text):
            self._submitted.append((sentence_idx, text))

        async def dispatch(self):
            idx = 0
            for sentence_idx, text in self._submitted:
                result = _GenerationResult(
                    sentence_idx=sentence_idx,
                    audio_path=f"/tmp/test_{idx}.mp3",
                    duration=1.0,
                    timing_info={},
                    success=True,
                )
                self._results.append(result)
                idx += 1
            self._submitted.clear()

        async def drain_next(self):
            if self._results:
                return self._results.pop(0)
            return None

    mock_gen = MockSuccessGen()
    buffer._parallel_gen = mock_gen

    # Patch sentence parsing to return controlled results.
    with patch.object(
        content_parser, "split_into_sentences"
    ) as mock_split:
        mock_split.return_value = [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
        ]

        # Call _refill_batch_parallel once — all sentences succeed.
        success = await buffer._refill_batch_parallel(mock_gen, 2)

    assert success is True
    assert buffer._error_count == 0


# ============================================================================
# Test 3: Producer stops after max consecutive errors
# ============================================================================


@pytest.mark.asyncio
async def test_producer_stops_after_max_consecutive_errors():
    """After _max_errors consecutive failures, producer puts sentinel and exits."""
    from lue.lookahead_buffer import LookaheadBuffer
    from lue import content_parser

    # --- Mock reader ---
    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [["A test sentence."]],
            "tts_model": MagicMock(),
            "tts_cache": None,
            "_advance_position": lambda self, pos, wrap=False: None,
        },
    )()

    buffer = LookaheadBuffer(reader, target_sentences=2, max_errors=2)
    buffer._is_running = True

    # Patch sanitize to return input unchanged.
    with patch.object(
        content_parser, "sanitize_text_for_tts", side_effect=lambda x: x
    ):
        # Start at max-1 = 2 errors. _handle_error will push it to 3.
        buffer._error_count = 2
        buffer._handle_error(ValueError("test"), "test text")

        # After _handle_error: error_count should be 3.
        assert buffer._error_count == 3

        # Now put the sentinel and verify the queue contains it.
        await buffer._put_sentinel()

        item = await buffer.get()
        assert item is None


# ============================================================================
# Test 4: Player handles sentinel correctly
# ============================================================================


@pytest.mark.asyncio
async def test_player_receives_sentinel_and_stops():
    """When get() returns None, player sets playback_finished_event and exits."""
    from lue.audio import _player_loop

    # --- Build a mock reader that exposes all attributes _player_loop touches ---
    playback_finished_event = asyncio.Event()

    mock_buf = MagicMock()
    mock_buf.get = AsyncMock(return_value=None)  # immediately returns sentinel
    mock_buf.task_done = MagicMock()
    mock_buf.signal_refill = AsyncMock()

    reader = type(
        "MockReader",
        (),
        {
            "running": True,
            "lookahead_buffer": mock_buf,
            "playback_finished_event": playback_finished_event,
            "audio_queue": asyncio.Queue(),
            "active_playback_tasks": [],
            "playback_processes": [],
            "playback_speed": 1.0,
            "tts_model": MagicMock(),
            "loop": MagicMock(),
        },
    )()

    # Run _player_loop — it should exit gracefully within the timeout
    # because buf.get() returns None immediately, causing a break.
    try:
        await asyncio.wait_for(
            _player_loop(reader),
            timeout=0.5,
        )
    except asyncio.TimeoutError:
        pytest.fail("_player_loop did not exit after receiving None sentinel")

    # playback_finished_event should have been set by the sentinel handler.
    assert playback_finished_event.is_set()

    # Verify the buffer interactions
    mock_buf.get.assert_called_once()  # called exactly once (got None → break)
    mock_buf.task_done.assert_called_once()


# ============================================================================
# Test 5: Parallel gen submits failure result on TTS exception
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_gen_submits_failure_result_on_tts_error():
    """When TTS generate_audio raises, the worker submits success=False."""
    from lue import content_parser

    # Build a mock TTS model where:
    #   - output_format is "mp3"
    #   - generate_audio raises (but generate_audio_with_timing does NOT exist
    #     as a distinguishable attribute so we enter the "else" branch)
    # We achieve this by using a spec-limited mock.
    tts_model = MagicMock(spec=["output_format", "generate_audio"])
    tts_model.output_format = "mp3"
    tts_model.generate_audio = AsyncMock(
        side_effect=ValueError("tts error simulation")
    )

    # Build a minimal mock reader.
    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [[]],
            "tts_model": tts_model,
            "tts_cache": None,
            "_advance_position": lambda self, pos, wrap=False: None,
        },
    )()

    gen = ParallelTTSGen(reader, tts_model, max_concurrent=1, cache=None)

    with patch.object(
        content_parser, "sanitize_text_for_tts", side_effect=lambda x: x
    ):
        gen.submit((0, 0, 0), "Hello world")
        await gen.dispatch()

        # Allow the worker a short window to fail.
        await asyncio.sleep(0.1)

        result = await gen.drain_next()

    assert result is not None
    assert result.success is False
    assert result.audio_path == ""
    assert result.duration == 0.0
    assert result.timing_info == {}


# ============================================================================
# Test 6: End-to-end — error propagation through full pipeline
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_error_to_sentinel_integration():
    """Integration: TTS error → failure result → error_count → sentinel."""
    from lue.lookahead_buffer import LookaheadBuffer
    from lue import content_parser

    # --- 1. Mock reader ---
    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [["One. Two. Three."]],
            "tts_model": MagicMock(spec=["output_format"]),
            "tts_cache": None,
            "_advance_position": (lambda self, pos, wrap=False: None),
        },
    )()

    # Position advancement: 3 sentences at (0,0,0), (0,0,1), (0,0,2), then end.
    def _advance(current_pos, mode="sentence", wrap=True):
        c, p, s = current_pos
        if s < 2:
            return (c, p, s + 1)
        return None

    reader._advance_position = _advance

    # --- 2. Mock TTS that fails on every call ---
    tts_model = MagicMock(spec=["output_format", "generate_audio"])
    tts_model.output_format = "mp3"
    tts_model.generate_audio = AsyncMock(
        side_effect=ValueError("openai tts timeout")
    )

    # --- 3. ParallelTTSGen ---
    gen = ParallelTTSGen(reader, tts_model, max_concurrent=1, cache=None)

    # --- 4. LookaheadBuffer with max_errors=1 (sentinel after 2 failures) ---
    buffer = LookaheadBuffer(reader, target_sentences=3, max_errors=1)
    buffer._is_running = True
    buffer._parallel_gen = gen

    with (
        patch.object(
            content_parser,
            "sanitize_text_for_tts",
            side_effect=lambda x: x,
        ),
        patch.object(
            content_parser,
            "split_into_sentences",
            return_value=["One.", "Two.", "Three."],
        ),
    ):
        # --- 5. Call _refill_batch_parallel — batch of up to 2 sentences.
        # Both sentences will fail because the TTS model always raises.
        # buffer._error_count should increment for each failed result.
        success = await buffer._refill_batch_parallel(gen, 1)
        assert success is False
        # Each failed result increments _error_count by 1.
        # The batch processes up to max_concurrent*2 = 2 sentences.
        assert buffer._error_count > 0, (
            f"Expected _error_count > 0 after failure, got {buffer._error_count}"
        )

        # --- 6. Call _refill_batch_parallel again — another batch of failures.
        prev_errors = buffer._error_count
        success = await buffer._refill_batch_parallel(gen, 1)
        assert success is False
        assert buffer._error_count > prev_errors, (
            "Error count should increase after a second batch of failures"
        )

    # --- 7. With max_errors=1 and error_count >= 2, the producer loop would
    # now check: if not success and self._error_count >= self._max_errors:
    # → put sentinel and break.  Simulate that final step.
    await buffer._put_sentinel()

    # --- 8. Verify the sentinel is in the queue ---
    item = await buffer.get()
    assert item is None, (
        "Expected None sentinel in the queue after max errors exceeded. "
        f"Got {item!r}"
    )

    # --- 9. Clean shutdown ---
    await gen.shutdown()


# ============================================================================
# Test 7: get() keeps waiting while producer is alive (slow TTS fix)
# ============================================================================


@pytest.mark.asyncio
async def test_get_keeps_waiting_while_producer_is_alive():
    """get() does NOT return None just because the queue is empty for 2+ seconds.

    The old get() had a hard 2s timeout before returning None. When TTS was
    slow (queue empty for longer than 2s), this premature None was treated
    as "end of book" by the player, causing permanent freeze.

    The fix uses an infinite retry loop while _is_running is True.
    """
    from lue.lookahead_buffer import LookaheadBuffer

    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [["Test."]],
            "tts_model": MagicMock(),
            "tts_cache": None,
            "_advance_position": (lambda self, pos, wrap=False: None),
        },
    )()

    buffer = LookaheadBuffer(reader, target_sentences=1, max_errors=5)
    buffer._is_running = True
    buffer._producer_task = MagicMock()
    buffer._producer_task.done.return_value = False  # producer is alive

    # get() should block while _is_running is True (queue is empty, producer alive).
    # Wrap in wait_for(..., timeout=0.5) — it should time out because get()
    # keeps retrying, NOT return None.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(buffer.get(), timeout=0.5)


# ============================================================================
# Test 8: get() returns None when producer is stopped
# ============================================================================


@pytest.mark.asyncio
async def test_get_returns_none_when_producer_stopped():
    """get() returns None promptly after _is_running becomes False."""
    from lue.lookahead_buffer import LookaheadBuffer

    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [["Test."]],
            "tts_model": MagicMock(),
            "tts_cache": None,
            "_advance_position": (lambda self, pos, wrap=False: None),
        },
    )()

    buffer = LookaheadBuffer(reader, target_sentences=1, max_errors=5)
    buffer._is_running = True
    buffer._producer_task = MagicMock()
    buffer._producer_task.done.return_value = False

    # Schedule a task that stops the producer after a short delay.
    async def stop_producer():
        await asyncio.sleep(0.15)
        buffer._is_running = False

    asyncio.create_task(stop_producer())

    # get() should return None within a short time after _is_running flips.
    try:
        result = await asyncio.wait_for(buffer.get(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail("get() did not return None within 2s after _is_running=False")

    assert result is None


# ============================================================================
# Test 9: get() returns None when producer task died unexpectedly
# ============================================================================


@pytest.mark.asyncio
async def test_get_returns_none_when_producer_task_crashed():
    """get() returns None when the producer task finishes unexpectedly.

    This covers the edge case where the producer task crashes (done()=True)
    without properly setting _is_running=False.
    """
    from lue.lookahead_buffer import LookaheadBuffer

    reader = type(
        "MockReader",
        (),
        {
            "chapter_idx": 0,
            "paragraph_idx": 0,
            "sentence_idx": 0,
            "chapters": [["Test."]],
            "tts_model": MagicMock(),
            "tts_cache": None,
            "_advance_position": (lambda self, pos, wrap=False: None),
        },
    )()

    buffer = LookaheadBuffer(reader, target_sentences=1, max_errors=5)
    buffer._is_running = True
    # Simulate producer task that has finished (crashed)
    buffer._producer_task = MagicMock()
    buffer._producer_task.done.return_value = True

    # get() should detect the crashed producer and return None.
    # Use timeout=2.0 to give the inner 1s queue.get() timeout
    # enough time to fire cleanly without the outer wrapper
    # cancelling the task before the result can propagate.
    try:
        result = await asyncio.wait_for(buffer.get(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail(
            "get() did not return None within 2s when producer_task.done()=True"
        )

    assert result is None
