"""Unit tests for the Speechify TTS provider.

Covers initialization gating (API key / dependency), request payload shape,
the base64 + ``speech_marks`` response contract (including ms→s timing
conversion), text chunking, single-request timing, and timeout / retry
behaviour.
"""

from __future__ import annotations

import asyncio
import base64
import random
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from lue.tts.speechify_tts import SpeechifyAPIError, SpeechifyTTS


# ── Helpers ────────────────────────────────────────────────────────────────

def make_response(
    status: int = 200,
    content: bytes = b"",
    content_type: str = "application/json",
    json_data: dict | None = None,
    headers: dict | None = None,
):
    """Build a mock httpx-like response."""
    resp = Mock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    if headers:
        resp.headers.update(headers)
    resp.content = content
    resp.text = "" if json_data is None else str(json_data)
    resp.json = Mock(return_value=json_data if json_data is not None else {})
    resp.request = Mock()
    return resp


def speech_payload(audio: bytes = b"MP3DATA", speech_marks: dict | None = None) -> dict:
    """Build a successful /v1/audio/speech JSON body."""
    if speech_marks is None:
        speech_marks = {
            "type": "sentence",
            "start": 0,
            "end": 11,
            "start_time": 0,
            "end_time": 1500,
            "value": "Hello world",
            "chunks": [
                {"type": "word", "start": 0, "end": 5,
                 "start_time": 0, "end_time": 500, "value": "Hello"},
                {"type": "word", "start": 6, "end": 11,
                 "start_time": 500, "end_time": 1500, "value": "world"},
            ],
        }
    return {
        "audio_data": base64.b64encode(audio).decode(),
        "audio_format": "mp3",
        "speech_marks": speech_marks,
        "billable_characters_count": 11,
    }


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tts(mock_console: Mock) -> SpeechifyTTS:
    """Return an initialised SpeechifyTTS instance with a mock console."""
    instance = SpeechifyTTS(mock_console)
    instance.initialized = True
    instance.client = AsyncMock()
    return instance


# ── Identity / capability ──────────────────────────────────────────────────

class TestIdentity:
    def test_name_matches_filename(self, mock_console):
        assert SpeechifyTTS(mock_console).name == "speechify"

    def test_output_format(self, mock_console):
        assert SpeechifyTTS(mock_console).output_format == "mp3"

    def test_supports_word_timing(self, mock_console):
        """Speechify returns speech_marks, so word highlighting stays on."""
        assert SpeechifyTTS(mock_console).supports_word_timing is True

    def test_default_voice_is_wren(self, mock_console):
        assert SpeechifyTTS(mock_console).voice == "wren"

    def test_explicit_voice_overrides_default(self, mock_console):
        assert SpeechifyTTS(mock_console, voice="george").voice == "george"

    def test_default_model(self, mock_console):
        assert SpeechifyTTS(mock_console)._model == "simba-3.2"

    def test_model_env_override(self, mock_console, monkeypatch):
        monkeypatch.setenv("LUE_SPEECHIFY_TTS_MODEL", "simba-3.0")
        assert SpeechifyTTS(mock_console)._model == "simba-3.0"


# ── _should_retry ──────────────────────────────────────────────────────────

class TestShouldRetry:
    def test_timeouts_retry(self):
        assert SpeechifyTTS._should_retry(asyncio.TimeoutError()) is True
        assert SpeechifyTTS._should_retry(TimeoutError()) is True

    def test_httpx_errors_retry(self):
        for name in (
            "TimeoutException", "ConnectError", "ConnectTimeout",
            "ReadTimeout", "WriteTimeout", "PoolTimeout",
            "RemoteProtocolError", "NetworkError",
        ):
            exc_class = type(name, (Exception,), {})
            assert SpeechifyTTS._should_retry(exc_class()) is True, name

    def test_connection_errors_retry(self):
        assert SpeechifyTTS._should_retry(ConnectionError()) is True
        assert SpeechifyTTS._should_retry(BrokenPipeError()) is True

    @pytest.mark.parametrize(
        "status,expected",
        [(429, True), (500, True), (502, True), (503, True),
         (400, False), (401, False), (404, False), (422, False)],
    )
    def test_status_code_branch(self, status, expected):
        assert SpeechifyTTS._should_retry(
            SpeechifyAPIError("x", status_code=status)
        ) is expected

    @pytest.mark.parametrize("status,expected", [(429, True), (503, True), (400, False)])
    def test_http_status_error_branch(self, status, expected):
        HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        exc = HTTPStatusError()
        exc.response = MagicMock()
        exc.response.response = MagicMock()
        exc.response.status_code = status
        assert SpeechifyTTS._should_retry(exc) is expected

    def test_generic_errors_do_not_retry(self):
        assert SpeechifyTTS._should_retry(ValueError()) is False
        assert SpeechifyTTS._should_retry(RuntimeError()) is False


# ── Chunking ───────────────────────────────────────────────────────────────

class TestChunking:
    def test_short_text_single_chunk(self):
        assert SpeechifyTTS._chunk_text("hello", 2000) == ["hello"]

    def test_over_limit_hard_split_concatenates(self):
        text = "A" * 12000
        chunks = SpeechifyTTS._chunk_text(text, 2000)
        assert all(len(c) <= 2000 for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_sentence_boundaries(self):
        text = ("This is sentence number %d and it is long enough to matter. " % 0) * 200
        chunks = SpeechifyTTS._chunk_text(text, 2000)
        assert all(len(c) <= 2000 for c in chunks)
        assert len(chunks) > 1
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith((".", "!", "?"))


# ── speech_marks → timing conversion ───────────────────────────────────────

class TestExtractTimings:
    def test_milliseconds_converted_to_seconds(self):
        marks = {
            "type": "sentence",
            "chunks": [
                {"type": "word", "value": "Hello", "start_time": 0, "end_time": 500},
                {"type": "word", "value": "world", "start_time": 500, "end_time": 1500},
            ],
        }
        assert SpeechifyTTS._extract_timings(marks) == [
            ("Hello", 0.0, 0.5),
            ("world", 0.5, 1.5),
        ]

    def test_list_of_sentence_entries(self):
        marks = [
            {"chunks": [{"type": "word", "value": "Hi",
                         "start_time": 0, "end_time": 250}]},
            {"chunks": [{"type": "word", "value": "there",
                         "start_time": 300, "end_time": 800}]},
        ]
        assert SpeechifyTTS._extract_timings(marks) == [
            ("Hi", 0.0, 0.25),
            ("there", 0.3, 0.8),
        ]

    def test_merged_word_chunks_preserved(self):
        """Chunks may merge source words; create_word_mapping reconciles."""
        marks = {"chunks": [
            {"type": "word", "value": "Dr. Smith", "start_time": 0, "end_time": 1280},
        ]}
        assert SpeechifyTTS._extract_timings(marks) == [("Dr. Smith", 0.0, 1.28)]

    def test_non_word_chunks_skipped(self):
        marks = {"chunks": [
            {"type": "sentence", "value": "whole", "start_time": 0, "end_time": 900},
            {"type": "word", "value": "ok", "start_time": 0, "end_time": 100},
        ]}
        assert SpeechifyTTS._extract_timings(marks) == [("ok", 0.0, 0.1)]

    @pytest.mark.parametrize("marks", [None, {}, {"chunks": []}])
    def test_empty_marks(self, marks):
        assert SpeechifyTTS._extract_timings(marks) == []

    def test_chunk_missing_fields_skipped(self):
        marks = {"chunks": [
            {"type": "word", "value": "no-times"},
            {"type": "word", "start_time": 0, "end_time": 100},
        ]}
        assert SpeechifyTTS._extract_timings(marks) == []


# ── Offset anchoring (the highlighting fix) ─────────────────────────────────
#
# Speechify merges neighbours into one chunk ("October 2001"), which used to
# send create_word_mapping down its greedy fallback and shift the alignment
# permanently after the first merge. These tests pin the re-anchoring onto
# lue's own word boundaries.

def _chunk(value, c0, c1, t0_ms, t1_ms, ctype="word"):
    return {"type": ctype, "value": value, "start": c0, "end": c1,
            "start_time": t0_ms, "end_time": t1_ms}


class TestOffsetAnchoring:
    def test_merged_chunk_splits_back_into_source_words(self):
        """One chunk covering two source words yields two separate timings."""
        text = "October 2001 was good."
        marks = {"chunks": [
            _chunk("October 2001", 0, 12, 0, 2000),
            _chunk("was", 13, 16, 2000, 2300),
            _chunk("good.", 17, 22, 2300, 3000),
        ]}

        timings = SpeechifyTTS._extract_timings(marks, text)

        assert [w for w, _, _ in timings] == ["October", "2001", "was", "good"]
        assert timings[0] == ("October", 0.0, pytest.approx(7 / 12 * 2.0))
        assert timings[1] == ("2001", pytest.approx(8 / 12 * 2.0), 2.0)
        assert timings[2] == ("was", 2.0, 2.3)
        # End uses the stripped core ("good", 4 chars) inside the [17,22] chunk,
        # so 2.3 + 0.7 * 4/5.
        assert timings[3] == ("good", 2.3, pytest.approx(2.86))

    def test_split_chunks_merge_back_into_one_source_word(self):
        """Speechify splitting a word ('C'+'ROSSING') still yields one word."""
        text = "Crossing the Chasm"
        marks = {"chunks": [
            _chunk("C", 0, 1, 0, 300),
            _chunk("ROSSING", 1, 8, 300, 1500),
            _chunk("the", 9, 12, 1500, 1800),
            _chunk("Chasm", 13, 18, 1800, 2500),
        ]}

        timings = SpeechifyTTS._extract_timings(marks, text)

        assert [w for w, _, _ in timings] == ["Crossing", "the", "Chasm"]
        assert timings[0] == ("Crossing", 0.0, 1.5)

    def test_output_is_one_to_one_with_timing_calculator(self, mock_console):
        """The whole point: the anchored list must match the tokenizer exactly,
        so create_word_mapping short-circuits on a perfect match."""
        from lue.timing_calculator import _get_highlightable_words, create_word_mapping

        text = "October 2001 ISBN 0-06-018987-8 The original hardcover edition."
        # Chunks deliberately merge across several source words each.
        marks = {"chunks": [
            _chunk("October 2001", 0, 12, 0, 1500),
            _chunk("ISBN 0-06-018987-8", 13, 32, 1500, 4000),
            _chunk("The", 33, 36, 4000, 4200),
            _chunk("original", 37, 45, 4200, 5000),
            _chunk("hardcover", 46, 55, 5000, 5900),
            _chunk("edition.", 56, 64, 5900, 6700),
        ]}

        timings = SpeechifyTTS._extract_timings(marks, text)
        words = _get_highlightable_words(text)

        assert [w for w, _, _ in timings] == words
        assert create_word_mapping(words, timings) == list(range(len(words)))

    def test_pure_punctuation_tokens_not_emitted(self):
        """Standalone '™' is not spoken and the reader does not count it, so
        the anchored list must not count it either (else every later index is
        off by one)."""
        text = "PerfectBound ™ and the logo."
        marks = {"chunks": [
            _chunk("PerfectBound ™", 0, 14, 0, 2000),
            _chunk("and", 15, 18, 2000, 2300),
            _chunk("the", 19, 22, 2300, 2600),
            _chunk("logo.", 23, 28, 2600, 3200),
        ]}

        timings = SpeechifyTTS._extract_timings(marks, text)

        assert [w for w, _, _ in timings] == ["PerfectBound", "and", "the", "logo"]

    def test_falls_back_when_offsets_absent(self):
        """No character offsets at all -> legacy chunk segmentation."""
        marks = {"chunks": [
            {"type": "word", "value": "hello", "start_time": 0, "end_time": 500},
            {"type": "word", "value": "world", "start_time": 500, "end_time": 1500},
        ]}
        assert SpeechifyTTS._extract_timings(marks, "hello world") == [
            ("hello", 0.0, 0.5), ("world", 0.5, 1.5)
        ]

    def test_falls_back_when_source_text_absent(self):
        """Offsets present but nothing to anchor against -> legacy path."""
        marks = {"chunks": [_chunk("hello", 0, 5, 0, 500)]}
        assert SpeechifyTTS._extract_timings(marks, "") == [("hello", 0.0, 0.5)]

    def test_time_at_interpolates_inside_chunk(self):
        anchors = [(0, 10, 0.0, 2.0)]
        assert SpeechifyTTS._time_at(0, anchors) == 0.0
        assert SpeechifyTTS._time_at(5, anchors) == 1.0
        assert SpeechifyTTS._time_at(10, anchors) == 2.0

    def test_time_at_clamps_before_first_and_after_last(self):
        anchors = [(10, 20, 1.0, 2.0)]
        assert SpeechifyTTS._time_at(0, anchors) == 1.0
        assert SpeechifyTTS._time_at(99, anchors) == 2.0

    def test_time_at_interpolates_across_gap(self):
        """Between two chunks time is interpolated, never jumped."""
        anchors = [(0, 4, 0.0, 1.0), (8, 12, 3.0, 4.0)]
        # position 6 sits in the gap (4 -> 8), i.e. halfway between 1.0 and 3.0
        assert SpeechifyTTS._time_at(6, anchors) == 2.0

    def test_end_never_precedes_start(self):
        """Malformed (inverted) offsets still yield a monotonic pair."""
        text = "abc def"
        marks = {"chunks": [
            _chunk("abc def", 0, 7, 2000, 1000),
        ]}
        timings = SpeechifyTTS._extract_timings(marks, text)
        assert all(end >= start for _, start, end in timings)


# ── generate_audio ─────────────────────────────────────────────────────────

class TestGenerateAudio:
    @pytest.mark.asyncio
    async def test_base64_decoded_and_written(self, tts, tmp_path):
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"ID3AUDIO"))
        )

        await tts.generate_audio("Hello world", out)

        with open(out, "rb") as f:
            assert f.read() == b"ID3AUDIO"

    @pytest.mark.asyncio
    async def test_request_payload_shape(self, tts, tmp_path):
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload())
        )

        await tts.generate_audio("Hello", out)

        _, kwargs = tts.client.post.call_args
        payload = kwargs["json"]
        assert payload["input"] == "Hello"
        assert payload["voice_id"] == "wren"
        assert payload["audio_format"] == "mp3"
        assert payload["model"] == "simba-3.2"
        assert "options" in payload
        assert "text_normalization" in payload["options"]

    @pytest.mark.asyncio
    async def test_endpoint_url(self, tts, tmp_path):
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload())
        )

        await tts.generate_audio("Hello", out)

        args, _ = tts.client.post.call_args
        assert args[0] == "https://api.speechify.ai/v1/audio/speech"

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self, tts, tmp_path):
        with pytest.raises(ValueError):
            await tts.generate_audio("   ", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self, mock_console, tmp_path):
        instance = SpeechifyTTS(mock_console)
        with pytest.raises(RuntimeError):
            await instance.generate_audio("Hello", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_missing_audio_data_raises(self, tts, tmp_path):
        tts.client.post = AsyncMock(
            return_value=make_response(json_data={"speech_marks": {}})
        )
        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_undecodable_audio_data_raises(self, tts, tmp_path):
        tts.client.post = AsyncMock(
            return_value=make_response(
                json_data={"audio_data": "!!!not-base64!!!"}
            )
        )
        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))

    @pytest.mark.asyncio
    async def test_long_text_chunked_and_concatenated(self, tts, tmp_path):
        """Text over 2000 chars is split into multiple requests."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"SEG"))
        )

        await tts.generate_audio("A" * 12000, out)

        assert tts.client.post.await_count == 6  # 6 × 2000
        with open(out, "rb") as f:
            assert f.read() == b"SEG" * 6

    @pytest.mark.asyncio
    async def test_chunked_timings_offset(self, tts, tmp_path):
        """Second chunk's timings are shifted by the first chunk's duration."""
        out = str(tmp_path / "out.mp3")
        marks = {
            "chunks": [
                {"type": "word", "value": "one",
                 "start_time": 0, "end_time": 1000},
            ]
        }
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"SEG", marks))
        )

        timings = await tts.get_raw_timing_data("A" * 4500, out)

        assert tts.client.post.await_count == 3
        # Chunk 1 ends at 1.0s (+0.15 padding) → chunk 2 starts there, etc.
        assert [t[1] for t in timings] == [0.0, 1.15, 2.3]


# ── Error handling ─────────────────────────────────────────────────────────

class TestErrorResponse:
    def test_error_message_extracted(self):
        resp = make_response(
            status=400,
            json_data={"error": {
                "code": "validation_failed",
                "message": "Field input must not exceed 2000 characters",
            }},
        )
        msg = SpeechifyTTS._error_message(resp)
        assert "400" in msg
        assert "must not exceed 2000 characters" in msg

    @pytest.mark.parametrize("status", [401, 404, 429])
    def test_error_raises_with_status(self, mock_console, status):
        resp = make_response(status=status, json_data={"error": {"message": "nope"}})
        with pytest.raises(SpeechifyAPIError) as exc:
            SpeechifyTTS(mock_console)._parse_response(resp, "hi")
        assert exc.value.status_code == status

    def test_retry_after_parsed(self):
        resp = make_response(status=429, json_data={}, headers={"retry-after": "2.5"})
        assert SpeechifyTTS._retry_after(resp) == 2.5

    def test_retry_after_absent(self):
        assert SpeechifyTTS._retry_after(make_response(status=429)) is None

    def test_retry_after_garbage(self):
        resp = make_response(status=429, json_data={}, headers={"retry-after": "soon"})
        assert SpeechifyTTS._retry_after(resp) is None

    def test_raw_audio_body_accepted(self, mock_console):
        """A non-JSON 200 (raw audio) is passed through as bytes."""
        resp = make_response(content_type="audio/mpeg", content=b"RAWDATA")
        audio, timings = SpeechifyTTS(mock_console)._parse_response(resp, "hi")
        assert audio == b"RAWDATA"
        assert timings == []


# ── Timing path ────────────────────────────────────────────────────────────

class TestTimingPath:
    @pytest.mark.asyncio
    async def test_raw_timing_data_writes_audio(self, tts, tmp_path):
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"ID3"))
        )

        timings = await tts.get_raw_timing_data("Hello world", out)

        assert timings == [("Hello", 0.0, 0.5), ("world", 0.5, 1.5)]
        with open(out, "rb") as f:
            assert f.read() == b"ID3"

    @pytest.mark.asyncio
    async def test_generate_audio_with_timing_single_request(self, tts, tmp_path):
        """Timing + audio must come from ONE billed request."""
        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"ID3"))
        )

        info = await tts.generate_audio_with_timing("Hello world", out)

        assert tts.client.post.await_count == 1
        assert sorted(info.keys()) == [
            "speech_duration", "total_duration", "word_mapping", "word_timings",
        ]
        assert info["word_timings"][0][0] == "Hello"
        assert info["total_duration"] > 0

    @pytest.mark.asyncio
    async def test_duration_fallback_when_ffprobe_fails(self, tts, tmp_path, monkeypatch):
        """Unreadable audio falls back to the last word's end time."""
        from lue import audio as lue_audio

        async def no_duration(path):
            return None

        monkeypatch.setattr(lue_audio, "get_audio_duration", no_duration)

        out = str(tmp_path / "out.mp3")
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"ID3"))
        )

        info = await tts.generate_audio_with_timing("Hello world", out)

        assert info["total_duration"] == pytest.approx(1.5, abs=0.001)


# ── Retry behaviour ────────────────────────────────────────────────────────

class TestRetryBehaviour:
    @pytest.mark.asyncio
    async def test_retry_then_success(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_RETRY_BASE_DELAY", 0.01)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise SpeechifyAPIError("concurrency", status_code=429)
            return make_response(json_data=speech_payload(b"OK"))

        tts.client.post = AsyncMock(side_effect=side_effect)

        await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_MAX_RETRIES", 2)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_RETRY_BASE_DELAY", 0.01)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        tts.client.post = AsyncMock(
            side_effect=SpeechifyAPIError("server error", status_code=503)
        )

        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert tts.client.post.await_count == 3  # attempts 0, 1, 2

    @pytest.mark.asyncio
    async def test_non_retryable_401_raises_immediately(self, tts, tmp_path):
        tts.client.post = AsyncMock(
            side_effect=SpeechifyAPIError("Unauthorized", status_code=401)
        )

        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert tts.client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_404_unknown_voice_raises(self, tts, tmp_path):
        tts.client.post = AsyncMock(
            side_effect=SpeechifyAPIError("Voice not found", status_code=404)
        )

        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        assert tts.client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout_wraps_request(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_MAX_RETRIES", 0)

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(10)
            return make_response()

        tts.client.post = AsyncMock(side_effect=slow_post)

        start = asyncio.get_event_loop().time()
        with pytest.raises(asyncio.TimeoutError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 3, f"timeout not enforced (took {elapsed:.1f}s)"

    @pytest.mark.asyncio
    async def test_backoff_schedule(self, tts, tmp_path, monkeypatch):
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_MAX_RETRIES", 3)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_RETRY_BASE_DELAY", 0.1)

        tts.client.post = AsyncMock(
            side_effect=SpeechifyAPIError("server error", status_code=500)
        )

        sleeps: list[float] = []

        async def mock_sleep(duration):
            sleeps.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "random", Mock(return_value=0.5))  # jitter = 0

        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))

        assert len(sleeps) == 3
        assert sleeps[0] == pytest.approx(0.1, abs=0.001)
        assert sleeps[1] == pytest.approx(0.2, abs=0.001)
        assert sleeps[2] == pytest.approx(0.4, abs=0.001)

    @pytest.mark.asyncio
    async def test_retry_after_extends_backoff(self, tts, tmp_path, monkeypatch):
        """A server Retry-After longer than our schedule wins."""
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_TIMEOUT", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_MAX_RETRIES", 1)
        monkeypatch.setattr("lue.config.SPEECHIFY_TTS_RETRY_BASE_DELAY", 0.1)

        tts.client.post = AsyncMock(
            side_effect=SpeechifyAPIError("rate limited", status_code=429,
                                          retry_after=0.75)
        )

        sleeps: list[float] = []

        async def mock_sleep(duration):
            sleeps.append(duration)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(random, "random", Mock(return_value=0.5))

        with pytest.raises(SpeechifyAPIError):
            await tts.generate_audio("Hello", str(tmp_path / "out.mp3"))

        assert sleeps[0] == pytest.approx(0.75, abs=0.001)


# ── Concurrency guard ──────────────────────────────────────────────────────

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_default_single_concurrency(self, mock_console):
        """Free plan allows 1 in-flight request; semaphore defaults to 1."""
        from lue import config
        instance = SpeechifyTTS(mock_console)
        assert config.SPEECHIFY_TTS_MAX_CONCURRENT == 1
        assert instance._semaphore._value == 1

    @pytest.mark.asyncio
    async def test_requests_serialised(self, tts, tmp_path):
        """Only one request may be in flight at a time."""
        in_flight = {"now": 0, "max": 0}

        async def slow_post(*args, **kwargs):
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            await asyncio.sleep(0.05)
            in_flight["now"] -= 1
            return make_response(json_data=speech_payload(b"SEG"))

        tts.client.post = AsyncMock(side_effect=slow_post)

        paths = [str(tmp_path / f"seg{i}.mp3") for i in range(3)]

        await asyncio.gather(
            *[tts.generate_audio("Hello there", p) for p in paths]
        )

        assert in_flight["max"] == 1, "concurrency limit not enforced"

    @pytest.mark.asyncio
    async def test_requests_queue_not_fail_when_bursting(self, tts, tmp_path):
        """A burst of requests all succeed (queued), none get a 429."""
        tts.client.post = AsyncMock(
            return_value=make_response(json_data=speech_payload(b"SEG"))
        )

        paths = [str(tmp_path / f"b{i}.mp3") for i in range(4)]
        await asyncio.gather(*[tts.generate_audio("Hello", p) for p in paths])

        for p in paths:
            with open(p, "rb") as f:
                assert f.read() == b"SEG"


# ── initialize ─────────────────────────────────────────────────────────────

class TestInitialize:
    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_console, monkeypatch):
        monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
        instance = SpeechifyTTS(mock_console)
        assert await instance.initialize() is False
        assert instance.initialized is False

    @pytest.mark.asyncio
    async def test_success(self, mock_console, monkeypatch):
        import httpx

        monkeypatch.setenv("SPEECHIFY_API_KEY", "sk_test")
        instance = SpeechifyTTS(mock_console)

        mock_client = MagicMock(spec=httpx.AsyncClient)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: mock_client)

        assert await instance.initialize() is True
        assert instance.initialized is True
        assert instance.client is mock_client

    @pytest.mark.asyncio
    async def test_api_key_sent_as_bearer(self, mock_console, monkeypatch):
        import httpx

        monkeypatch.setenv("SPEECHIFY_API_KEY", "sk_test")
        instance = SpeechifyTTS(mock_console)

        captured: dict = {}

        def fake_client(**kwargs):
            captured.update(kwargs)
            return MagicMock(spec=httpx.AsyncClient)

        monkeypatch.setattr(httpx, "AsyncClient", fake_client)
        assert await instance.initialize() is True
        assert captured["headers"]["Authorization"] == "Bearer sk_test"
        assert captured["headers"]["Content-Type"] == "application/json"


# ── warm-up ────────────────────────────────────────────────────────────────

class TestWarmUp:
    @pytest.mark.asyncio
    async def test_warm_up_generates_and_cleans_up(self, tts, tmp_path, monkeypatch):
        from lue import config

        monkeypatch.setattr(config, "AUDIO_DATA_DIR", str(tmp_path))

        tts.generate_audio = AsyncMock()
        await tts.warm_up()

        tts.generate_audio.assert_awaited_once()
        args, _ = tts.generate_audio.call_args
        assert args[0] == "Ready."
        # Warm-up file is removed afterwards (generate_audio is mocked, so
        # explicitly create the file to verify cleanup).
        warmup_file = tmp_path / f".warmup_speechify.{tts.output_format}"
        warmup_file.write_bytes(b"stub")
        await tts.warm_up()
        assert not warmup_file.exists()
