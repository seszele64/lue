# Spec v2 — Gapless Persistent-Sink Audio Architecture

- **Status:** Proposed (v2) — supersedes the rejected v1 ("Request changes").
- **Change:** `gapless-persistent-sink`
- **Depends on:** `add-openai-tts` (provider capability reporting), existing `timing_calculator`.
- **Scope:** Replace the per-sentence `ffplay <file>` process with **one long-lived
  ffplay sink** fed raw PCM over stdin by a paced writer, with decode-side `atempo`,
  an authoritative audio clock, and a generation-token restart protocol.
- **Non-goals:** real-time pitch-preserving timestretch beyond `atempo`; gapless
  *streaming* TTS (synthesis latency is unchanged); audio crossfading (removed — see ADR-1).

This document is implementable on its own: it fixes exact FFmpeg/ffplay command lines,
clock equations, config constants, CLI validation, state inventory, file-level
migration, a restart state machine, and a verification harness with real fixtures.

---

## 0. Resolution matrix (reviewer P0/P1 → section)

| ID | Reviewer finding | Fix location | Status |
|----|------------------|--------------|--------|
| P0-1 | Highlight/word sync regress by SINK_LEAD 400 ms + crossfade 200 ms; wall-clock anchor `reader.py:1437-1466`, `_word_update_loop` `1139-1233`, `elapsed*speed` | §4 (Audio clock), ADR-1 | Resolved (single audio-clock domain; crossfade deleted) |
| P0-2 | Restart/generation-token vs in-flight pipe bytes undefined | §5 (Generation & restart) | Resolved (explicit token lifecycle + 10-step teardown) |
| P0-3 | Speed/`atempo` placement missing; `audio.py:279-296`, `326-350`, `__main__.py:203-206` unclamped | §6 (Speed) | Resolved (decode-side chain order + clamp) |
| P0-4 | Decode/resample/trim correctness; Kokoro WAV vs MP3; `duration is None` skip `audio.py:262-264` | §7 (Decode) | Resolved (exact cmd; no skip; byte-derived duration) |
| P0-5 | Starvation/underrun + backpressure deadlock; `MAX_QUEUE_SIZE=4` items vs 64 KB pipe | §8 (Flow control) | Resolved (byte/ms bounds, silence-pad, drain timeouts) |
| P1-1 | Overlap clamp contradiction (kokoro 0.6, gtts 0.1, default 0.4 vs [150,250] ms; `test_gtts_tts.py:56`) | §9.1, ADR-1 | Resolved (overlap removed; config + test migrated) |
| P1-2 | `pkill` scoping too broad (`input_handler.py:296-307`) | §9.2 | Resolved (PID kill; tagged fallback) |
| P1-3 | Pause overshoot | §9.3 | Resolved (clock-snapshot resume; no sleeps) |
| P1-4 | Temp file lifecycle | §9.4 | Resolved (per-generation session dir; delete-after-decode) |
| P1-5 | Migration/rollback flag | §9.5 | Resolved (`LUE_PERSISTENT_SINK`, `--legacy-audio`) |
| — | Full state inventory missed prior | §10 | Closed (complete table) |

---

## 1. Goals, invariants, non-goals

**Goals**
- G1 Audible playback is continuously fed: no silence between consecutive sentences
  caused by process spawn/decode (the current per-file `ffplay` gap).
- G2 Word/sentence highlighting is driven by an **audio clock** that models the
  sink's audible position, not by wall-clock at command dispatch.
- G3 Any navigation/pause/speed change/seek/book switch tears the pipeline down and
  recreates it deterministically, with no stray processes, no leaked fds/files, and
  no writes into a dead pipe.
- G4 Slow/absent TTS never deadlocks the writer and never crashes; it degrades to
  bounded silence.
- G5 The whole feature can be toggled off to the legacy per-file path for rollout.

**Invariants (must hold at all times)**
- I1 One *current* generation integer. Every loop checks it after each await.
- I2 At most one sink process and one writer task per generation.
- I3 The writer never blocks permanently: every `await` on pipe I/O is bounded by a
  timeout or by cancellation from teardown.
- I4 All PCM handed to the sink is `s16le / 48000 Hz / 2 ch`; byte count maps
  linearly to stream seconds at `BYTES_PER_SEC`.
- I5 No task touches `sink`/`writer`/`decoder` state after its generation is invalid.
- I6 The stream position `P(t)` is monotonic non-decreasing and never stalls in
  steady state (silence is inserted rather than starving).

**Non-goals**
- NG1 Crossfading sentences (removed; ADR-1).
- NG2 Reading ffplay's internal playhead (not portable); we model it (§4).
- NG3 Changing TTS synthesis; providers keep writing files.

---

## 2. Architecture overview

```
 producer_task (per generation)
   │  TTS: tts_model.generate_audio_with_timing(text, path)
   │  emit DecodeJob(file, pos, timing_info, gen)
   ▼
 decode_queue  (bounded: MAX_QUEUED_SENTENCES + PCM_RING_MAX_MS estimate)
   │
   ▼
 writer_task (per generation)  ── owns sink stdin, pacing, clock, segment table
   │  decode Job → ffmpeg (atempo,aresample,aformat) → s16le chunks
   │  append Segment / extend out_duration; write paced chunks; pad silence on starvation
   ▼
 sink.stdin  ──pipe(64 KB)──▶  sink_process = persistent ffplay (-autoexit)
   │
   ▼
 sink_clock (SinkClock) ── P(t) ──▶ _word_update_loop (20 Hz)
   └─ SegmentTable  ──▶ sentence/word highlight + reader position while PLAYING

 player_task (sink supervisor, per generation)
   └─ await sink.wait(); on EOF+close → playback_finished_event.set()
```

Removed: rotating `AUDIO_BUFFERS` per-sentence `ffplay`, per-item `asyncio.sleep`,
`active_playback_tasks`, `playback_processes` population, `_new_sentence_started`
as a wall-clock anchor.

Added modules:
- `lue/audio_sink.py` — `SinkProcess`, `SinkClock`, `Segment`, `SegmentTable`,
  `writer_task`, `decode_job`.
- Reuse `lue/timing_calculator.py` unchanged (word mapping stays source-domain).

---

## 3. Component contracts

- **SinkProcess**: wraps `ffplay`; exposes `.proc`, `.stdin` (StreamWriter),
  `.pid`, `.generation`. `start()`, `close_stdin()`, `terminate()`, `kill()`,
  `wait()`. Never raises on already-dead process.
- **DecodeJob**: `(generation, file_path, chapter, paragraph, sentence,
  timing_info, requested_speed)`.
- **Segment**: table row (see §4.3). `generation`, `out_start`, `out_duration`
  (grows while writing), `src_duration`, `src_offset`, `speed`, `word_timings`,
  `word_mapping`, `is_silence`, `pos`.
- **SegmentTable**: `open_segment(seg)`, `extend(bytes)`, `close_segment()`,
  `find(P) -> Segment`, `clear()`, `discard_generation(gen)`.
- **SinkClock**: `position(now) -> float`, `anchor(t_first_write)`, `reset()`,
  `calibration` (`SINK_LEAD`). Owned by the writer's generation.

---

## 4. P0-1 — Audio clock (the sync fix)

### 4.1 Root cause of v1
v1 set `current_word_start_time = loop.time()` when the sentence command was
dispatched (`reader.py:1458`) and `_word_update_loop` compared
`(loop.time() - start) * speed` against source-domain `word_timings`
(`reader.py:1150-1152`). The sink introduces (a) `SINK_LEAD` before any written
byte is audible and (b) a crossfade that advances the next sentence's onset. Both
shift the true audible onset away from the dispatch time, so the highlight leads by
`SINK_LEAD` and is additionally perturbed by the crossfade (net error ≈ 200–600 ms).
The dispatch-time anchor is therefore **deleted**.

### 4.2 PCM format and derived constants
- Sink format: `s16le`, `48000 Hz`, `2 ch` (channel-interleaved).
- `BYTES_PER_SEC = 48000 * 2 * 2 = 192000`.
- Pipe capacity on Linux `= 65536 B = 0.3413 s`.

### 4.3 The audio clock (authoritative)

Output timeline = the sink's audible position, in **post-atempo output seconds**.

```
anchor:            t_play_start = t_first_write + SINK_LEAD
position:          P(t)         = clamp(t - t_play_start, 0, P_end)
```

- `t_first_write` = monotonic `loop.time()` captured by the writer immediately
  before the **first** `write()` of the generation.
- `SINK_LEAD` = measured constant (§4.5) absorbing pipe fill + ffplay input buffer
  + device latency. **It is in the device/wall-clock domain and does NOT scale with
  playback speed** (latency is content-independent; speed is handled in the mapping).
- `P(t)` is valid because writing is *paced to realtime* (≥ 1 chunk per
  `WRITER_CHUNK_MS`) and the writer **never lets the sink starve** (§8): any gap is
  filled with silence that is itself part of the stream, so `P(t)` keeps advancing.
- Silence inserted for starvation is recorded as a silence Segment; `P` therefore
  stays consistent with what is audible.

**Per-segment mapping** (strictly one generation; `seg.generation == current`):

```
local_out = P - seg.out_start                     # output seconds into segment
local_src = seg.src_offset + local_out * seg.speed # back to source/TTS seconds
word_idx  = find_word(seg.word_timings, local_src) # same algorithm as today
```

- `seg.speed` is the generation's `playback_speed` (speed changes restart; §6).
- `seg.src_offset` = source-seconds already skipped (0 unless resumed mid-sentence; §9.3).
- `seg.out_duration` grows as bytes are written; measured as
  `bytes_written_for_seg / BYTES_PER_SEC`. Correctness does **not** depend on
  `ffprobe` (this also removes the P0-4 "duration None" dependency, §7.4).
- Answer to "what does duration mean post-crossfade/atempo":
  `out_duration = src_duration / speed` and `speech_duration`/`word_timings`
  remain **source-domain**; conversion happens only in `local_src`.
- Answer to "when does `_new_sentence_started` fire relative to audible onset":
  it is **retired as an anchor**. The visible sentence advances at
  `P(t) >= seg.out_start` (audible onset), computed by `_word_update_loop`
  polling `SinkClock`; there is no dispatch-time event to mis-time.

### 4.4 Word index algorithm (replaces `reader.py:1155-1223`)
Given `seg.word_timings` (source-domain, already continuity-adjusted by
`timing_calculator`) and `seg.word_mapping`:
1. If `local_src` falls in `[start_i, end_i)` → TTS word `i`.
2. Else if `local_src >= max_end` (and `local_out < out_duration`) → hold last word.
3. Map TTS word `i` through `word_mapping`; if several originals map to `i`,
   subdivide `[start_i,end_i)` equally (existing behavior, `reader.py:1189-1202`).
4. If no `word_timings` → equal split `time_per_word = src_duration/num_words`
   (existing fallback, `reader.py:1220-1223`).
5. If the current segment is a silence segment → clear highlight (`ui_word_idx`
   left at last value; do not advance position into a silence segment's words).

### 4.5 Calibration of `SINK_LEAD`
- Default `SINK_LEAD_MS = 250` (`= SINK_PRIME_MS(150) + SINK_DEVICE_LATENCY_MS(100)`),
  overridable via `LUE_SINK_LEAD_MS`.
- Measured end-to-end by the harness (§14): emit a leading impulse fixture, timestamp
  first write `t_first_write`, detect audible onset with the dummy/timestamping sink,
  `SINK_LEAD = t_audible - t_first_write`; assert `|SINK_LEAD - configured| <= 120 ms`.
- If `SINK_LEAD` cannot be measured (no device), the configured value is used and the
  sync test switches to the modeled sink (byte-timestamped), which is exact.

---

## 5. P0-2 — Generation token & restart protocol

### 5.1 Token lifecycle
- `reader.audio_generation: int` starts at 0; incremented on **every** teardown.
- `reader.sink_clock`, `reader.segment_table` are rebuilt per generation and tagged.
- Each loop captures `gen = reader.audio_generation` at spawn and re-checks at the
  top of every iteration and after every `await`:
  `if gen != reader.audio_generation: return`.
- `_word_update_loop` ignores segments with `generation != reader.audio_generation`.
- All async cancellation is wrapped to tolerate `CancelledError`/`TimeoutError`.

### 5.2 Teardown / restart primitive (single source of truth)
`stop_and_clear_audio(reader)` (replaces `audio.py:41-94`) runs these steps **in order**;
`_restart_audio_after_navigation`, `_handle_pause_toggle`, `_switch_book`, `_shutdown`
all call it:

```
 1. reader.audio_generation += 1                      # invalidate everything
 2. cancel + await producer_task                      # bounded 1.0 s
 3. cancel + await writer_task                         # may be inside drain(); CancelledError ok
 4. cancel + await all decoder tasks                   # usually none (decoder runs inside writer)
 5. sink.close_stdin()                                 # tolerate BrokenPipe/ConnectionReset
 6. sink.terminate(); wait 0.5 s; if alive sink.kill(); wait 0.2 s
 7. cancel + await player_task (supervisor)            # usually already returned via proc.wait
 8. drain decode_queue and legacy audio_queue (get_nowait + task_done)
 9. reset: sink=None, sink_clock=None, table.clear(), producer/player/writer=None,
          playback_finished_event.clear()
    (the next start spawns a fresh sink, which creates a fresh stdin pipe; the
     PCM pipe is never reused across generations)
10. best-effort delete session temp dir (§9.4)
```

Ordering rationale: cancel writers **before** closing stdin (so a blocked `drain()`
is unwound), kill the sink **before** cancelling its supervisor (so `proc.wait()`
returns and the supervisor exits), and close stdin **before** SIGTERM (so a healthy
sink flushes and `-autoexit` can exit on EOF).

### 5.3 In-flight pipe bytes / error handling
- Writer catches `BrokenPipeError`, `ConnectionResetError`, `ConnectionAbortedError`
  around every `write()`/`drain()`; on any, it stops writing and returns (teardown
  owns cleanup). It never retries against a dead pipe.
- `drain()` is wrapped with `SINK_WRITE_TIMEOUT_S = 5.0`; on timeout → treat as
  stalled sink, abort writer, and request restart (never hang).
- ffplay early-exit: if `sink.proc.returncode is not None` while the writer still has
  data, the writer aborts; the supervisor sets `playback_finished_event` with a
  warning and (if not an intended EOF) posts a restart.
- Decoder failure: see §7.5.

### 5.4 Interaction with existing call sites
- `_restart_audio_after_navigation` (`reader.py:955-973`): keep `audio_restart_lock`;
  inside, call `stop_and_clear_audio` then `play_from_current_position`.
- `_kill_audio_immediately` (`input_handler.py:296-307`): rewrite (§9.2); it only
  bumps the token and kills by PID (non-blocking). The follow-up navigation command
  performs the full 10-step teardown under the lock.
- `_handle_pause_toggle` (`reader.py:1073-1089`): acquires `pause_toggle_lock`, then
  `audio_restart_lock`, then `stop_and_clear_audio`; snapshots position for resume (§9.3).
- `_shutdown` (`reader.py:1235-1256`): add `writer_task`, `sink`, and any `decoder`
  to the cleanup; call `stop_and_clear_audio`; then existing UI/terminal restore.
- `_switch_book` (`reader.py:101-107`): `stop_and_clear_audio` already bumps the token
  and resets the clock/table; additionally reset `playback_finished_event`.
- `finish` (`reader.py:1492-1496`): semantics below.
- **Lock order (mandatory):** `pause_toggle_lock` → `audio_restart_lock`. No path takes
  them in the reverse order.

### 5.5 `finish` / bytes-written vs bytes-played
- Producer hits end-of-book → puts `EOF_SENTINEL` on `decode_queue`.
- Writer finishes the current job, flushes remaining buffered PCM, then
  `sink.close_stdin()`.
- ffplay `-autoexit` plays out the remaining buffered PCM and exits at EOF.
- Supervisor (`player_task`) does `await sink.wait()`; only **then** sets
  `playback_finished_event`. This event therefore means *bytes played*, not
  *bytes written*.
- `finish` handler:
  ```
  if player_task and not done: await playback_finished_event.wait()
  await audio_queue.join()          # legacy drain, no-op in sink mode
  break
  ```
- If the sink does not exit within `SINK_EOF_TIMEOUT_S = 3.0` after `close_stdin`,
  SIGTERM→KILL and set the event with a warning (bounded failure, never a hang).
- `playback_finished_event` is **cleared** at every generation start (fixes the
  current latent bug where it is never cleared after the first finish).

### 5.6 Pipeline state machine (text)

```
                 (re)start()
   IDLE ─────────────────────────────▶ PRIMING
    ▲                                    │  sink spawned; writer fills
    │                                    │  SINK_PRIME_MS before anchor
    │        resume()                    ▼
    │   ┌─────────── PAUSED ◀── PLAYING ──┐
    │   │            │  ▲        │  │    │
    │   │   stop     │  │ resume │  │ starvation detected
    │   │            ▼  │        │  ▼
    │   │          (teardown)    │  PADDING (silence inserted; stays PLAYING)
    │   │                       EOF_SENTINEL
    │   │                        │
    │   └────────────────────────┼──────▶ DRAINING
    │                            │          │ stdin closed; sink drains
    │                            │          ▼
    └────────── NAV/QUIT ◀──── RESTARTING ◀─ FINISHED (event set)

 Rows: IDLE, PRIMING, PLAYING, PADDING(a substate of PLAYING), PAUSED,
       DRAINING, FINISHED, RESTARTING.  Any state → RESTARTING on nav/pause/seek/
       speed-change/book-switch; RESTARTING → PRIMING or IDLE.
```

| From | To | Trigger | Guard/action |
|------|----|---------|--------------|
| IDLE | PRIMING | `play_from_current_position` | gen stable; sink spawn ok |
| PRIMING | PLAYING | first PCM written + `SINK_PRIME_MS` | `anchor(t_first_write)` |
| PLAYING | PADDING | PCM ring < `PCM_LOW_WATER_MS` and producer idle | append silence seg |
| PADDING | PLAYING | next DecodeJob decoded | close silence seg |
| PLAYING/PADDING | DRAINING | `EOF_SENTINEL` | writer flush + close stdin |
| DRAINING | FINISHED | `sink.wait()` returns | `playback_finished_event.set()` |
| PLAYING | PAUSED | `pause` | snapshot P; stop sink |
| PAUSED | PRIMING | `pause` again | resume with `src_offset` |
| any | RESTARTING | nav/seek/speed/book | `gen += 1`; teardown |
| RESTARTING | PRIMING/IDLE | teardown done | if not paused: start |
| any | IDLE | `_shutdown` | teardown; no restart |

---

## 6. P0-3 — Speed and `atempo`

### 6.1 Placement (decision)
Apply speed **decode-side** in the per-sentence ffmpeg transcode, so the sink always
receives nominal-rate PCM and needs no reconfiguration on speed change. The previous
per-file `ffplay -af atempo...` (`audio.py:279-296`) is removed.

### 6.2 Filter chain (exact order, exact string)
```
-filter:a "<atempo_chain>,aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo"
```
- `atempo` first (source sample-rate domain; preserves pitch), then `aresample`
  (to sink rate, band-limited), then `aformat` (s16 + stereo).
- `atempo_chain` construction (speed `s` already clamped):
  ```
  if s == 1.0:  chain = ""            # omit atempo entirely
  else:
      parts = []; r = s
      while r > 2.0: parts.append("atempo=2.0"); r /= 2.0
      while r < 0.5: parts.append("atempo=0.5"); r /= 0.5
      if abs(r - 1.0) > 1e-4: parts.append(f"atempo={r:.4f}")
      chain = ",".join(parts) + ","
  ```
- With `SPEED_MAX = 4.0`, the worst chain is `atempo=2.0,atempo=2.0,`.

### 6.3 Clamp & validation
- `SPEED_MIN = 0.5`, `SPEED_MAX = 4.0`, `SPEED_STEP = 0.1`, default `1.0`.
- CLI `--speed` (`__main__.py:200-206, 311-312`): if outside `[SPEED_MIN, SPEED_MAX]`
  → print error and `sys.exit(2)` (strict), *before* constructing the reader.
- Runtime `_increase_speed`/`_decrease_speed`: clamp to the same range and trigger a
  restart only if the value changed.
- Speed changes take effect by **restart** from the current sentence start, so all
  segments in a generation share one `speed`; `SinkClock` never sees a mixed tempo.

### 6.4 Relationship to overlap (P1-1)
Crossfade is removed (ADR-1), so there is no "does L scale with speed?" ambiguity.
`SINK_LEAD` is wall-clock/device and speed-independent; `seg.speed` is applied only
in the source↔output mapping. The `atempo` value is computed once per generation from
the clamped speed.

---

## 7. P0-4 — Decode / resample / trim correctness

### 7.1 Provider output facts (as implemented)
| Provider | `output_format` | Source rate | Notes |
|----------|-----------------|-------------|-------|
| kokoro | `wav` | 24000 Hz mono | `soundfile` write, PCM |
| edge | `mp3` | ~24000 Hz mono | MP3 has encoder delay/padding |
| openai | `mp3` | 24000 Hz mono | `response_format="mp3"` |
| gtts | `mp3` | ~24000 Hz mono | MP3 |
| capcut | `mp3` | unknown | validated by `_looks_like_mp3` |

The decoder must accept both WAV and MP3 and arbitrary source rates.

### 7.2 Exact decode command (per sentence)
```
ffmpeg -nostdin -hide_banner -loglevel error \
  -i <src_file> -map 0:a:0 \
  -filter:a "<atempo_chain>aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo" \
    # <atempo_chain> already ends in a comma when non-empty, and is the empty
    # string at speed 1.0, so the concatenation yields a valid filtergraph.
  -ar 48000 -ac 2 -f s16le pipe:1
```
- `-nostdin`: never let ffmpeg read the application's terminal stdin (the sink owns
  its own stdin pipe; this prevents fd contention).
- `-map 0:a:0`: first audio stream; a missing audio stream is a decode failure (§7.5).
- `-f s16le pipe:1`: raw PCM on stdout, read as an async stream, chunked at
  `WRITER_CHUNK_MS`.
- WAV (24 kHz mono) and MP3 (any rate) both pass through the same chain.
- Resume mid-sentence adds `-ss <src_offset>` **before** `-i` (input seek; §9.3).

### 7.3 Trim policy
- **No `silenceremove` by default.** `silenceremove` is opt-in only
  (`--trim-silence`, default off) because of clipping/over-trim risk; thresholds when
  enabled: `silenceremove=start_periods=1:start_duration=0.30:start_threshold=-45dB:stop_periods=-1:stop_duration=0.30:stop_threshold=-45dB`.
- MP3 encoder delay/padding adds ≈ 20–100 ms of near-silence; accepted, not trimmed.
- Leading silence in TTS output is accepted; concatenation keeps it gapless.

### 7.4 Duration derivation (removes the `duration is None` skip)
- **Remove** `if duration is None or duration <= 0: task_done(); continue`
  (`audio.py:262-264`). A sentence is never silently dropped at the player.
- Authoritative duration is byte-derived at the writer:
  `out_duration = written_bytes_for_segment / BYTES_PER_SEC`.
- `src_duration = out_duration * speed`; `word_timings` are source-domain and used
  directly in `local_src`.
- `get_audio_duration` (`audio.py:98-105`) is retained for **logging/validation only**
  (e.g., detect padding vs speech); a `None` result no longer affects playback.
- Fallbacks when there are no `word_timings`: use `src_duration` for equal-split
  highlight; if `src_duration` is 0/unknown, skip highlight for that sentence but
  still play it.

### 7.5 Decode-failure fallback
1. ffmpeg non-zero exit, or zero bytes produced, or `BrokenPipe` while decoding:
   retry once with the same command.
2. Second failure → log `ERROR`, emit `DECODE_FAIL_SILENCE_MS = 500` ms of silence as
   a silence segment (keeps the timeline continuous and the sink alive), record the
   sentence as failed, and let the producer advance position normally.
3. Only if the writer itself is being torn down do we abort (generation check).

---

## 8. P0-5 — Flow control: starvation, underrun, backpressure

### 8.1 Bounds are byte/ms-based, not item-count
- `MAX_QUEUE_SIZE = 4` **items** is replaced by two limits:
  - `MAX_QUEUED_SENTENCES = 8` (hard cap; bounds restart latency and memory).
  - `PCM_RING_MAX_MS = 2000` → `PCM_RING_MAX_BYTES = 384000` (decoded PCM between
    producer/writer and the pipe).
- Pipe is 64 KB (0.341 s at 192000 B/s); `PCM_RING_MAX_MS` is deliberately ≥ 2× pipe
  so a burst decode can keep the pipe full while the sink drains.

### 8.2 Writer protocol
- Writer consumes DecodeJobs, runs ffmpeg, and writes output in
  `WRITER_CHUNK_MS = 20` ms chunks (3840 B) to `sink.stdin`.
- Pacing: after each chunk the writer ensures it has written at most
  `PCM_RING_MAX_MS` ahead of `P(t)`; it awaits `drain()` when the pipe is full.
- **Starvation/underrun handling = silence-pad, never starve:**
  - Low-water trigger: when the ready PCM < `PCM_LOW_WATER_MS = 300` and no DecodeJob
    output is immediately available, append a silence segment of
    `UNDERRUN_PAD_MS = 100` ms and write its bytes.
  - A single starvation episode is capped at `MAX_UNDERRUN_PAD_MS = 2000`; if exceeded,
    log a `WARNING` and continue padding (never crash). Total padded silence per
    episode is recorded for verification.
  - Why this beats "let it underrun": `P(t)` stays consistent (silence is content),
    the sink never exits early, and word mapping skips the silence segment cleanly.
- Backpressure: when `drain()` blocks, the writer stops reading ffmpeg stdout, ffmpeg
  blocks on write, and the producer blocks on `MAX_QUEUED_SENTENCES`/ring estimate.
  Deadlock is impossible because the sink always consumes; if it dies, `drain()`
  raises `BrokenPipe` and the writer aborts (§5.3).

### 8.3 EOF / flush on sentinel
- Producer end-of-book (or graceful error stop) puts `EOF_SENTINEL` on `decode_queue`.
- Writer: finish current job → flush buffered PCM → `sink.close_stdin()` →
  `await sink.stdin.wait_closed()` (tolerate errors).
- No `None` ever shares the PCM queue with bytes; the sentinel is a distinct object on
  the job queue, preserving ordering.

### 8.4 Bounded everyswhere
| Operation | Bound |
|-----------|-------|
| `sink.stdin.drain()` | `SINK_WRITE_TIMEOUT_S = 5.0` |
| `sink.terminate()→wait` | `SINK_TERM_TIMEOUT_S = 0.5` |
| `sink.kill()→wait` | `SINK_KILL_TIMEOUT_S = 0.2` |
| post-EOF sink exit | `SINK_EOF_TIMEOUT_S = 3.0` |
| task cancel/join | `TASK_CANCEL_TIMEOUT_S = 1.0` |
| decoder terminate | `DECODER_TERM_TIMEOUT_S = 0.3` |

---

## 9. P1 resolutions

### 9.1 P1-1 — overlap clamp contradiction
- **Decision:** remove audio overlap/crossfade entirely (ADR-1). The persistent sink is
  gapless by concatenation; overlap was only a hack for per-file process gaps.
- Migrate: `config.OVERLAP_SECONDS` → deprecated alias of `SENTENCE_GAP_SECONDS = 0.0`
  (silence inserted between sentences; default 0). `TTS_OVERLAP_SECONDS`
  (`config.py:27-30`, kokoro 0.6 / gtts 0.1) is deprecated and ignored; a one-time
  `WARNING` is logged if non-empty.
- `TTSBase.get_overlap_seconds` (`base.py:178-187`) returns `None`; `GTTTTS.get_overlap_seconds`
  (`gtts_tts.py:64-66`) returns `None`.
- `tests/test_gtts_tts.py:56` (`assert == 0.1`) is updated to assert `is None`
  (and a new test asserts the deprecation warning).
- No clamp range `[150,250] ms` exists in v2; this removes the contradiction.

### 9.2 P1-2 — `pkill` scoping
- Remove `pkill -f ffplay` (`input_handler.py:304`) and `pkill -9 -f 'ffplay.*buffer_'`
  (`audio.py:67`).
- Kill by tracked PID only: `os.kill(sink.pid, SIGKILL)` / `os.kill(decoder.pid, SIGKILL)`,
  tolerating `ProcessLookupError`.
- Fallback (only if a PID is unknown) uses an app-scoped tag: the sink is launched with
  `-window_title lue-audio-sink-<pid>-<gen>` and `pkill -f 'lue-audio-sink-'`; this never
  matches unrelated ffplay processes.
- `_kill_audio_immediately` is made non-blocking: it bumps `audio_generation`, kills by
  PID, and closes the stdin fd; it performs **no** `subprocess.run` on the event loop.

### 9.3 P1-3 — pause overshoot
- Overshoot in the current design comes from `await asyncio.sleep(actual_duration - overlap)`
  (`audio.py:350`) continuing across a pause/seek, plus dispatch-time anchoring.
- v2 removes per-sentence sleeps; playback timing is owned by the sink. Pause:
  1. On `pause`, snapshot `P_p = sink_clock.position(now)` and locate `seg = table.find(P_p)`.
  2. `resume_src = seg.src_offset + (P_p - seg.out_start) * seg.speed`.
  3. If `seg.is_silence` or `resume_src >= seg.src_duration - 0.05` → resume at the next
     sentence start (`resume_src = 0`, advance position).
  4. Persist `reader._resume = (pos, resume_src)`; teardown (`stop_and_clear_audio`).
  5. On resume, `play_from_current_position` starts decode with `-ss resume_src`, sets
     `seg.src_offset = resume_src`, `out_start = 0`, `anchor(now)` — highlight is exact.
- Pause latency budget: `PAUSE_MAX_LATENCY_MS = 150` (one `WRITER_CHUNK_MS` + kill).
  Verification asserts pause/resume position error `<= 1 word`.

### 9.4 P1-4 — temp file lifecycle
- Remove rotating `AUDIO_BUFFERS = buffer_0..5` (`config.py:107`) and their reuse.
- New per-generation session dir:
  `AUDIO_TMP_DIR = user_cache_dir("lue")/session-<pid>-<gen>/`.
- Each sentence writes `seg_<seq>.<fmt>` there; the writer deletes the file
  immediately after its decode completes (in `finally`), tolerating `OSError`.
- `stop_and_clear_audio` step 10 deletes the session dir (best-effort, retry 5×).
- `_shutdown` deletes the session dir too. Startup sweeps stale
  `session-*-*` dirs older than `AUDIO_TMP_MAX_AGE_S = 3600`.
- Provider warm-up files (`.warmup_*`) are unchanged.

### 9.5 P1-5 — migration/rollback flag
- `config.USE_PERSISTENT_SINK` (env `LUE_PERSISTENT_SINK`, default `1`).
- CLI `--legacy-audio` forces `USE_PERSISTENT_SINK = False`.
- Legacy path = the existing per-file `_producer_loop`/`_player_loop`, preserved behind
  the flag for one release; both paths share `stop_and_clear_audio` and state names.
- Telemetry/log line at startup states which path is active.

---

## 10. Full state inventory (v2)

| State | Type | Init | Owner | Set/used by | Cleared/teardown |
|-------|------|------|-------|-------------|------------------|
| `running` | bool | True | main | loops | `_shutdown` |
| `is_paused` | bool | False | main | pause handler | — |
| `playback_speed` | float | 1.0 | main | speed handlers | reset? no |
| `audio_generation` | int | 0 | main | every teardown; loops | never (monotonic) |
| `producer_task` | Task\|None | None | pipeline | `play_from_current_position` | teardown step 2 |
| `player_task` | Task\|None | None | pipeline | sink supervisor | teardown step 7 |
| `writer_task` | Task\|None | None | pipeline | feeds sink | teardown step 3 |
| `decoder_tasks` | list[Task] | [] | pipeline | transient decode | teardown step 4 |
| `sink` (SinkProcess) | obj\|None | None | pipeline | writer/teardown | teardown step 6/9 |
| `sink_clock` | SinkClock\|None | None | pipeline | `_word_update_loop` | step 9 |
| `segment_table` | SegmentTable | empty | pipeline | writer/word loop | step 9 |
| `decode_queue` | Queue | empty | pipeline | producer→writer | step 8 |
| `audio_queue` (legacy) | Queue | empty | legacy | legacy player | step 8 |
| `playback_finished_event` | Event | clear | pipeline | player supervisor | cleared step 9 + at start |
| `audio_restart_lock` | Lock | — | main | nav/pause/restart | never |
| `pause_toggle_lock` | Lock | — | main | pause handler | never |
| `current_pause_toggle_task` | Task\|None | None | main | pause handler | `_shutdown`, next toggle |
| `pending_restart_task` | Task\|None | None | main | nav | `_shutdown` |
| `active_playback_tasks` (legacy) | list | [] | legacy | legacy player | legacy |
| `playback_processes` (legacy) | list | [] | legacy | legacy kill | legacy |
| `ui_word_idx` | int | 0 | UI | word loop | on new segment |
| `current_sentence_words` | list\|None | None | word loop | word loop | per segment |
| `current_sentence_duration` | float\|None | None | word loop | word loop | per segment |
| `current_word_timings` | list\|None | None | word loop | word loop | per segment |
| `current_word_mapping` | list\|None | None | word loop | word loop | per segment |
| `current_word_start_time` | float | — | **retired** | — | replaced by `SinkClock` |
| `_resume` | tuple\|None | None | main | pause/resume | consumed |

Prior inventory omissions (now explicit): `producer_task`, `player_task`,
`playback_finished_event`, `pause_toggle_lock`, `current_pause_toggle_task`.

---

## 11. Config constants (values)

```python
# ── Persistent sink ─────────────────────────────────────────────
AUDIO_SINK_RATE          = 48000
AUDIO_SINK_CHANNELS      = 2
AUDIO_SINK_SAMPLE_FMT    = "s16le"
BYTES_PER_SEC            = 48000 * 2 * 2          # 192000
SINK_PRIME_MS            = 150
SINK_DEVICE_LATENCY_MS   = 100
SINK_LEAD_MS             = 250                    # env LUE_SINK_LEAD_MS; measured §4.5
SINK_LEAD                = SINK_LEAD_MS / 1000.0
SINK_WRITE_TIMEOUT_S     = 5.0
SINK_TERM_TIMEOUT_S      = 0.5
SINK_KILL_TIMEOUT_S      = 0.2
SINK_EOF_TIMEOUT_S       = 3.0
WRITER_CHUNK_MS          = 20                     # 3840 B/chunk
PCM_RING_MAX_MS          = 2000                   # 384000 B
PCM_LOW_WATER_MS         = 300
UNDERRUN_PAD_MS          = 100
MAX_UNDERRUN_PAD_MS      = 2000
MAX_QUEUED_SENTENCES     = 8
DECODE_FAIL_SILENCE_MS   = 500
DECODE_TERM_TIMEOUT_S    = 0.3
TASK_CANCEL_TIMEOUT_S    = 1.0

# ── Speed ───────────────────────────────────────────────────────
SPEED_MIN   = 0.5
SPEED_MAX   = 4.0
SPEED_STEP  = 0.1
DEFAULT_SPEED = 1.0

# ── Sentences / overlap (P1-1) ──────────────────────────────────
SENTENCE_GAP_SECONDS = 0.0
OVERLAP_SECONDS      = 0.0    # DEPRECATED alias; kept for config compat
TTS_OVERLAP_SECONDS  = {}     # DEPRECATED; values ignored + warning

# ── Temp lifecycle (P1-4) ───────────────────────────────────────
AUDIO_TMP_ROOT   = user_cache_dir("lue")
AUDIO_TMP_MAX_AGE_S = 3600

# ── Migration flag (P1-5) ───────────────────────────────────────
USE_PERSISTENT_SINK = env LUE_PERSISTENT_SINK (default 1)
```

---

## 12. CLI validation rules

| Flag | Rule | On violation |
|------|------|--------------|
| `--speed/-s` | parse float; require `SPEED_MIN <= v <= SPEED_MAX` | error message + `exit(2)` |
| `--over/-o` | deprecated; if provided warn "crossfade removed; ignored"; range `[0,1]` warned only | warn, continue |
| `--tts` | unchanged choices; `none` disables | argparse |
| `--legacy-audio` | new flag → `USE_PERSISTENT_SINK=False` | — |
| `--trim-silence` | new flag → enable conservative silenceremove (§7.3) | — |
| runtime `,`/`.` | clamp to `[SPEED_MIN, SPEED_MAX]`; restart only if changed | clamp |

Validation runs before `Lue(...)` is constructed and before `reader.playback_speed`
is assigned (`__main__.py:310-312`), so an invalid speed can never reach the atempo
builder.

---

## 13. File-level migration inventory

| File | Location | Change |
|------|----------|--------|
| `lue/audio.py` | `_producer_loop` 129-235 | Keep TTS generation; enqueue `DecodeJob` instead of `(file,dur,timing)`; byte/ms bound |
| `lue/audio.py` | `_player_loop` 237-361 | **Delete**; replaced by `writer_task` + `player_task` supervisor in `audio_sink.py` |
| `lue/audio.py` | atempo 275-296 | Move to decode-side ffmpeg chain (§6.2); **delete** ffplay `-af` |
| `lue/audio.py` | overlap 326-350, `actual_duration` 347-350 | **Delete**; no per-sentence sleep; overlap removed |
| `lue/audio.py` | `stop_and_clear_audio` 41-94 | Replace with 10-step teardown (§5.2); remove `pkill` 66-69 |
| `lue/audio.py` | `play_from_current_position` 107-127 | Spawn sink+writer+producer+supervisor; reset clock/table/event |
| `lue/audio.py` | `get_audio_duration` 98-105 | Retain for logging/validation only |
| `lue/reader.py` | `_initialize_state` 30-63 | Add `audio_generation`, `writer_task`, `sink`, `sink_clock`, `segment_table`, `decode_queue`, `_resume`; note retired fields |
| `lue/reader.py` | `_word_update_loop` 1139-1233 | Rewrite to `SinkClock` + `SegmentTable`; advance position at audible onset |
| `lue/reader.py` | `_new_sentence_started` 1437-1466 | Retire as anchor; remove `current_word_start_time`; segment table drives timing |
| `lue/reader.py` | `_shutdown` 1235-1256 | Cancel `writer_task`/sink; session dir cleanup |
| `lue/reader.py` | `finish` 1492-1496 | Event = sink-exit; clear event at start; bounded EOF wait |
| `lue/reader.py` | `_switch_book` 101-107 | Token bump + clock/table/event reset via `stop_and_clear_audio` |
| `lue/reader.py` | `_restart_audio_after_navigation` 955-973 | Use unified teardown under `audio_restart_lock` |
| `lue/reader.py` | `_handle_pause_toggle` 1073-1089 | Snapshot P; lock order; resume offset |
| `lue/input_handler.py` | `_kill_audio_immediately` 296-307 | PID kill + token bump; non-blocking; no `pkill`/`subprocess.run` |
| `lue/config.py` | 104-109 | New sink/speed constants; deprecate overlap; `AUDIO_TMP_ROOT`; flag |
| `lue/__main__.py` | 168-170, 200-206, 241-246, 310-312 | `--speed` clamp/validate; `--over` deprecation; `--legacy-audio`, `--trim-silence` |
| `lue/tts/base.py` | `get_overlap_seconds` 178-187 | Return `None` |
| `lue/tts/gtts_tts.py` | `get_overlap_seconds` 64-66 | Return `None` |
| `lue/tts/kokoro_tts.py` | — | No change (uses config via base) |
| `tests/test_gtts_tts.py` | line 56 | Assert `is None`; add deprecation-warning test |
| `lue/audio_sink.py` | **NEW** | `SinkProcess`, `SinkClock`, `Segment`, `SegmentTable`, `writer_task` |
| `lue/timing_calculator.py` | — | Unchanged; documented as source-domain |
| `tests/fixtures/audio/*` | **NEW** | Real fixtures (§14) |
| `tests/test_audio_sink_e2e.py` | **NEW** | Harness (§14) |

---

## 14. Verification harness & fixtures

### 14.1 Real fixtures (committed; generated from real providers)
Directory `tests/fixtures/audio/`:
- `kokoro_24k_mono_1s.wav` — real Kokoro output (offline, deterministic).
- `edge_24k_mono_1s.mp3` — real Edge TTS output.
- `openai_24k_mono_1s.mp3` — real OpenAI TTS output.
- `gtts_24k_mono_1s.mp3` — real gTTS output.
- `capcut_24k_mono_1s.mp3` — real CapCut output (if SDK available).
- `resample_44100_stereo_1s.wav` — 44.1 kHz stereo WAV (resample/format path).
- `mp3_padding_corner_1s.mp3` — short utterance to expose encoder delay/padding.
- `corrupt_not_audio.bin` — decode-failure path.
- `impulse_lead.wav` — sharp leading impulse for `SINK_LEAD` calibration.

Each fixture ships with `tests/fixtures/audio/manifest.json`:
`{file, provider, src_rate, channels, src_duration_s, text, word_timings:[{w,s,e}], sha256}`.
Generation script `tests/fixtures/audio/generate_fixtures.py` regenerates provider
fixtures; network/key steps are manual/CI-gated; Kokoro fixtures are offline.

### 14.2 Instrumented sink (deterministic CI)
- `SDL_AUDIODRIVER=dummy` for ffplay, **or** a fake sink script that reads `s16le`
  from stdin, sleeps `chunk/ BYTES_PER_SEC` to emulate a device clock, and logs
  `(recv_monotonic, byte_offset)` per chunk. Tests use the fake sink for exact timing
  assertions; one smoke test uses real ffplay+dummy driver to prove the command line.

### 14.3 Test cases (acceptance criteria)
| Test | Method | Pass criterion |
|------|--------|----------------|
| T1 gapless concat | 3 fixtures back-to-back, fake sink | byte range of segment n+1 starts exactly at `out_start(n)+out_duration(n)`; no unpadded silence |
| T2 lead modeling | `impulse_lead.wav`, fake sink timestamps | `|measured SINK_LEAD - configured| <= 120 ms` |
| T3 word sync | fixture with manifest timings; sample clock at 20 points | highlighted word == expected, ±1 word or ±80 ms |
| T4 atempo mapping | run 1.0x and 2.0x on same fixture | output duration ≈ src/speed ±5%; T3 still passes at 2.0x |
| T5 restart token | nav at t; assert old gen gone | old sink PID dead; `pgrep -f lue-audio-sink-` empty; new gen active; no fd leak |
| T6 BrokenPipe | kill sink mid-write | writer returns within `SINK_WRITE_TIMEOUT_S`; no unhandled exception; auto-restart |
| T7 underrun | throttle producer 2 s | writer inserts silence; total pad ≤ `MAX_UNDERRUN_PAD_MS`; clock stays monotonic; no crash |
| T8 backpressure | fast producer, real-time sink | `PCM_RING_MAX_MS` never exceeded; completes within timeout (no deadlock) |
| T9 pause/resume | pause mid-sentence, resume | position error ≤ 1 word; no overshoot; latency ≤ `PAUSE_MAX_LATENCY_MS` |
| T10 EOF/finish | single-sentence doc then `finish` | `playback_finished_event` set only after sink exit; all bytes played |
| T11 decode failure | `corrupt_not_audio.bin` | retry then 500 ms silence; producer advances; error logged; no gap crash |
| T12 resample/format | 44.1 kHz stereo WAV | sink receives 48 kHz stereo; duration within 1% |
| T13 temp lifecycle | run + restart + shutdown | session dir created, files deleted after decode, dir gone after teardown; no `buffer_*` files |
| T14 CLI validation | `--speed 0.2`, `--speed 5`, `--speed abc` | exit code 2 + message; `--speed 2.5` accepted |
| T15 legacy flag | `LUE_PERSISTENT_SINK=0` | per-file path runs; shared teardown works |

### 14.4 CI process hygiene checks
- Before/after each test: assert no `ffplay`/`ffmpeg` children of the test PID remain.
- Assert no open fds to the session dir.
- Assert `pgrep -f 'lue-audio-sink-'` returns nothing after teardown.
- All tests run with `SDL_AUDIODRIVER=dummy` unless marked `audio_device`.

---

## 15. Architecture Decision Records

### ADR-1 — Remove audio crossfade; gapless by PCM concatenation
- **Context:** v1 proposed a persistent sink *and* retained a 200 ms crossfade plus a
  configurable overlap (kokoro 0.6, gtts 0.1, default 0.4) that also had a
  contradictory `[150,250] ms` clamp. A single concatenating sink cannot crossfade
  without a mixer.
- **Decision:** Delete crossfade/overlap. Concatenate decoded PCM; optionally insert
  `SENTENCE_GAP_SECONDS` (default 0) silence.
- **Consequences (+):** Removes the clock-domain conflict (P0-1) and the clamp
  contradiction (P1-1); simpler, testable, deterministic. (−): no perceptual
  cross-smoothing; TTS already trims most sentence edge silence.
- **Alternatives rejected:** (a) software mixer in the writer — adds latency and CPU,
  and breaks `P(t)` linearity; (b) two overlapping sinks — reintroduces process gaps.

### ADR-2 — Decode-side `atempo` instead of per-playback `ffplay -af`
- **Context:** v1 kept `ffplay -af atempo` per file, making the sink apply speed per
  command and complicating restart.
- **Decision:** Apply `atempo` in the per-sentence ffmpeg decode chain; sink is
  tempo-agnostic.
- **Consequences (+):** Sink command fixed; speed changes restart cleanly; word
  mapping uses `local_src = out_local * speed`. (−): a re-decode on speed change
  (already required).

### ADR-3 — Model the clock; do not read ffplay's playhead
- **Context:** Need an authoritative audible position.
- **Decision:** `P(t)=clamp(t-t_first_write-SINK_LEAD)` with paced writing + silence
  padding so the model is exact; calibrate `SINK_LEAD` (§4.5).
- **Consequences (+):** portable, testable. (−): depends on the invariant that the
  sink never starves; enforced by §8.

### ADR-4 — Silence-pad on starvation
- **Decision:** Insert bounded silence rather than let the sink underrun.
- **Consequences (+):** `P(t)` stays linear, no early exit, no crash. (−): slow TTS
  yields a short silence (expected) rather than a shortened gap.

### ADR-5 — Byte/ms flow bounds, not item counts
- **Decision:** Bound by `PCM_RING_MAX_MS`/`MAX_QUEUED_SENTENCES`, not `MAX_QUEUE_SIZE`.
- **Consequences (+):** latency is bounded in time; long sentences cannot overrun the
  64 KB pipe. (−): slightly more bookkeeping.

---

## 16. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| `SINK_LEAD` varies by device | calibrate (§4.5); ±120 ms tolerated; model-based clock in CI |
| ffplay version buffer differences | format pinned (`-f s16le`), `-autoexit`; smoke test on real ffplay |
| MP3 padding shifts word sync | source-domain timings + ±80 ms tolerance; fixture `mp3_padding_corner_1s.mp3` |
| Writer blocked on `drain()` when sink dies | bounded `SINK_WRITE_TIMEOUT_S`; BrokenPipe catch; teardown |
| Lock-order deadlock (`pause` vs `nav`) | mandatory order `pause_toggle_lock`→`audio_restart_lock` |
| Temp dir growth | delete-after-decode; per-gen dir; startup sweep >1 h |
| Legacy path drift | shared teardown + state names; flag off for one release |

---

## 17. Open questions for the implementer

1. Preferred module boundary: keep `writer_task` in `audio.py` or a new
   `audio_sink.py` (spec assumes the latter)?
2. Is `--trim-silence` needed in this release, or defer (§7.3)?
3. Confirm `SDL_AUDIODRIVER=dummy` is available in CI for the smoke test.
4. Confirm the release window for removing `OVERLAP_SECONDS`/`TTS_OVERLAP_SECONDS`
   after the deprecation warning.
