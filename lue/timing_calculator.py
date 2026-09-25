"""
Word-level timing calculation module for the Lue eBook reader.
"""

import logging
import re
import string
from typing import List, Tuple, Optional, Dict, Any

def _sanitize_word(word: str) -> str:
    """
    Sanitize a word by stripping all non-alphanumeric characters and converting to lowercase.
    
    This function is used for case-insensitive word comparison by removing punctuation
    and other special characters, leaving only letters and numbers.
    
    Args:
        word: The word to sanitize
        
    Returns:
        Sanitized word containing only lowercase alphanumeric characters
    """
    if not word:
        return ""
    
    # Strip all non-alphanumeric characters
    sanitized = re.sub(r'[^a-zA-Z0-9]', '', word)
    
    # Convert to lowercase for case-insensitive comparison
    return sanitized.lower()


def _get_highlightable_words(text: str) -> list[str]:
    """
    Get list of words that should be considered for timing.
    
    This function filters out tokens that contain only punctuation/non-alphanumeric
    characters, which should not be counted as words for timing purposes.
    Strips punctuation (including commas from em dash replacement) for counting.
    
    Args:
        text: The text to process
        
    Returns:
        List of cleaned words that should be timed
    """
    # Split on whitespace to get tokens
    tokens = text.split()
    
    # Strip punctuation from tokens and filter out pure punctuation.
    #
    # A token only counts as a word if it contains at least one ASCII
    # alphanumeric character. That is exactly the test `reader` applies when it
    # builds `current_sentence_words` for highlighting, and the same test TTS
    # engines apply when deciding what to speak. Without it, standalone
    # non-ASCII punctuation ("™", "–", "•", "”") is counted here but not by the
    # reader, so every word *after* it resolves to an index one too high and
    # the highlight lands on the wrong word for the rest of the sentence.
    words = []
    for token in tokens:
        if not re.search(r'[a-zA-Z0-9]', token):
            continue
        cleaned = token.strip(string.punctuation)
        if cleaned:  # Only include non-empty cleaned tokens
            words.append(cleaned)
    
    return words


def _extract_core_word(token: str) -> str:
    """
    Extract the core word from a token by removing surrounding punctuation.
    
    This function is more robust than simple strip() as it handles nested
    punctuation and preserves internal punctuation like contractions.
    
    Args:
        token: The token to process
        
    Returns:
        The core word without surrounding punctuation
    """
    if not token:
        return token
    
    # Remove leading punctuation
    start = 0
    while start < len(token) and not token[start].isalnum():
        start += 1
    
    # Remove trailing punctuation
    end = len(token) - 1
    while end >= start and not token[end].isalnum():
        end -= 1
    
    if start <= end:
        return token[start:end + 1]
    else:
        return ""


def _align_sanitized(orig: List[str], tts: List[str]) -> Optional[List[int]]:
    """
    Exactly align two ``_sanitize_word`` sequences.

    The reader highlights tokens of the *original* sentence, but the TTS engine
    speaks ``content_parser.sanitize_text_for_tts``'s rewrite of it, which

    * splits hyphen- and dash-joined tokens into separate spoken words
      (``word-another`` -> ``word another``, ``10-20`` -> ``10 20``,
      ``past—it`` -> ``past it``) and
    * can let a single spoken word cover several tokens.

    Because both sides derive from the same string, the relationship is a
    fact, not a guess: one original token is exactly the concatenation of the
    spoken words it was split into. Walking both sequences by accumulated
    concatenation recovers it, whereas guessing word by word (see the fuzzy
    matcher below) drifts as soon as one token does not map 1:1 — a single
    ``one-size-fits-all`` shifts every later word of the sentence by three.

    Args:
        orig: ``_sanitize_word`` output for each original token
        tts: ``_sanitize_word`` output for each spoken word

    Returns:
        One TTS index per original token, or ``None`` when the sequences
        cannot be aligned (caller then falls back to fuzzy matching).
    """
    mapping: List[int] = []
    i = j = 0

    while i < len(orig) and j < len(tts):
        # Punctuation-only original token: no spoken word of its own.
        if not orig[i]:
            mapping.append(mapping[-1] if mapping else 0)
            i += 1
            continue
        # Punctuation-only spoken word: nothing highlights it on its own,
        # so leave it as an unmapped gap for the reader's owner lookup.
        if not tts[j]:
            j += 1
            continue

        # One original token split across several spoken words:
        # accumulate spoken words while they still spell out the token.
        acc, k = "", j
        while k < len(tts):
            cand = acc + tts[k]
            if not orig[i].startswith(cand):
                break
            acc = cand
            k += 1
            if acc == orig[i]:
                break
        if k > j and acc == orig[i]:
            mapping.append(j)
            i += 1
            j = k
            continue

        # One spoken word covering several original tokens.
        acc, k = "", i
        while k < len(orig):
            cand = acc + orig[k]
            if not tts[j].startswith(cand):
                break
            acc = cand
            k += 1
            if acc == tts[j]:
                break
        if k > i and acc == tts[j]:
            mapping.extend([j] * (k - i))
            i = k
            j += 1
            continue

        return None

    # Trailing unmatched entries on either side cannot be aligned.
    if i != len(orig) or j != len(tts):
        return None

    return mapping


def owner_word_index(word_mapping: List[int], tts_word_idx: int) -> int:
    """
    Resolve a spoken-word index to the original token that contains it.

    ``word_mapping`` maps each original token to *one* spoken word — the first
    of them when ``sanitize_text_for_tts`` split a hyphen- or dash-joined
    token (``word-another`` -> ``word another``). Every later spoken word of
    that token therefore has no entry of its own, and indexing the original
    token list with the spoken-word index (the old behaviour) lands one or
    more words ahead of what is actually being spoken.

    Args:
        word_mapping: original token index -> spoken word index
        tts_word_idx: spoken word currently being spoken

    Returns:
        Index of the original token owning that spoken word.
    """
    owners = [orig_idx for orig_idx, mapped in enumerate(word_mapping)
              if mapped <= tts_word_idx]
    return owners[-1] if owners else 0


def select_highlight_word(
    word_timings: Optional[List[Tuple[str, float, float]]],
    word_mapping: Optional[List[int]],
    elapsed: float,
    total_words: int,
    sentence_duration: float = 0.0,
) -> int:
    """
    Choose the token index to highlight at ``elapsed`` seconds of playback.

    Pure extraction of the body of ``reader.Lue._word_update_loop``, kept here
    so the selection can be tested without a running reader.

    Args:
        word_timings: (word, start, end) per spoken word, or ``None``/empty
        word_mapping: original token index -> spoken word index, if any
        elapsed: playback time adjusted for speed
        total_words: number of tokens in the displayed sentence
        sentence_duration: used to estimate timing when none is available

    Returns:
        Index into the displayed sentence's token list, clamped to range.
    """
    if total_words <= 0:
        return 0

    # No precise timings at all: distribute the sentence evenly.
    if not word_timings:
        if sentence_duration <= 0:
            return 0
        time_per_word = sentence_duration / total_words
        return min(int(elapsed / time_per_word), total_words - 1)

    # No mapping: trust the spoken word index (same index space assumption).
    if not word_mapping:
        for i, (_word, start_time, end_time) in enumerate(word_timings):
            if start_time <= elapsed < end_time:
                return min(i, total_words - 1)
        sentence_end = max(end for _w, _s, end in word_timings)
        if elapsed >= sentence_end:
            return total_words - 1
        return 0

    # Which spoken word is being uttered right now?
    tts_word_idx = None
    tts_word_start = tts_word_end = None
    for i, (_word, start_time, end_time) in enumerate(word_timings):
        if start_time <= elapsed < end_time:
            tts_word_idx, tts_word_start, tts_word_end = i, start_time, end_time
            break

    if tts_word_idx is None:
        sentence_end = max(end for _w, _s, end in word_timings)
        if elapsed >= sentence_end:
            tts_word_idx = len(word_timings) - 1
            _w, tts_word_start, tts_word_end = word_timings[tts_word_idx]

    if tts_word_idx is None:
        return 0

    mapped_orig_words = [orig_idx for orig_idx, mapped in enumerate(word_mapping)
                         if mapped == tts_word_idx]

    if mapped_orig_words:
        if len(mapped_orig_words) == 1:
            return min(mapped_orig_words[0], total_words - 1)
        # Several displayed tokens share one spoken word: split its duration
        # between them.
        tts_duration = tts_word_end - tts_word_start
        if tts_duration <= 0:
            return min(mapped_orig_words[0], total_words - 1)
        time_per_orig_word = tts_duration / len(mapped_orig_words)
        elapsed_in_tts_word = elapsed - tts_word_start
        sub_word_idx = min(int(elapsed_in_tts_word / time_per_orig_word),
                           len(mapped_orig_words) - 1)
        return min(mapped_orig_words[max(sub_word_idx, 0)], total_words - 1)

    # No displayed token owns this spoken word on its own:
    # sanitize_text_for_tts() splits hyphen- and dash-joined tokens
    # ("word-another", "past—it") into several spoken words, and only the
    # first carries the token's mapping entry. Stay on that token — indexing
    # the token list with the spoken word index landed the highlight ahead of
    # the word actually being spoken.
    return min(owner_word_index(word_mapping, tts_word_idx), total_words - 1)


def mapping_for_displayed_words(
    displayed_words: List[str],
    timing_info: Dict[str, Any],
) -> Optional[List[int]]:
    """
    Rebuild ``word_mapping`` for the sentence the reader is displaying.

    ``timing_info`` is produced from the text the TTS engine actually spoke,
    and ``content_parser.sanitize_text_for_tts`` rewrites that text before it
    is spoken: hyphen- and dash-joined tokens become separate words
    (``word-another`` -> ``word another``, ``past—it`` -> ``past it``,
    ``10-20`` -> ``10 20``). Its ``word_mapping`` therefore indexes the
    *spoken* token list, while ``reader.current_sentence_words`` and
    ``reader.ui_word_idx`` index the displayed sentence's tokens. Using it
    directly shifts the highlight by one position for every joined token
    earlier in the sentence — measured on real eBooks, 53% of ticks were on
    the wrong word for hyphenated sentences.

    Args:
        displayed_words: tokens of the sentence as rendered (the reader's
            ``current_sentence_words``)
        timing_info: timing dict returned by ``process_tts_timing_data``

    Returns:
        One timing index per displayed token, or ``None`` when no timings.
    """
    timings = timing_info.get("word_timings") or []
    if not timings:
        return None
    # Aligning the displayed tokens against the timings themselves resolves
    # both directions: tokens the engine split (several timings, one token)
    # and tokens it merged (one timing, several tokens).
    return create_word_mapping(displayed_words, timings) or timing_info.get("word_mapping")


def create_word_mapping(original_words: List[str], tts_word_timings: List[Tuple[str, float, float]]) -> Optional[List[int]]:
    """
    Create a mapping from original word indices to TTS word timing indices.
    This handles cases where TTS combines words or processes punctuation differently.
    
    Uses sanitized word comparison with fuzzy matching to handle:
    - Punctuation differences between original and TTS text
    - Case differences
    - Word combining/splitting by TTS engines
    - Punctuation-only tokens

    Before falling back to that fuzzy matching, the two sides are aligned by
    concatenation (``_align_sanitized``), which resolves the common case where
    ``sanitize_text_for_tts`` split a hyphen- or dash-joined token into
    several spoken words.
    
    Args:
        original_words: List of words from the original text
        tts_word_timings: List of (word, start_time, end_time) tuples from TTS
    
    Returns:
        List where index i contains the TTS timing index for original word i,
        or None if no valid mapping can be created.
    """
    # Handle edge cases: empty inputs
    if not original_words or not tts_word_timings:
        logging.debug("create_word_mapping: Empty input - original_words or tts_word_timings is empty")
        return None
    
    # Extract just the text from TTS timings
    tts_words = [word for word, _, _ in tts_word_timings]
    
    # Sanitize both original and TTS words for comparison
    orig_sanitized = [_sanitize_word(word) for word in original_words]
    tts_sanitized = [_sanitize_word(word) for word in tts_words]
    
    # If counts match and sanitized words match, create simple 1:1 mapping
    if len(original_words) == len(tts_words):
        # Check if sanitized versions match
        if orig_sanitized == tts_sanitized:
            logging.debug(f"create_word_mapping: Perfect 1:1 match with {len(original_words)} words")
            return list(range(len(original_words)))

    # Exact alignment first: recover split/merged tokens as a fact instead of
    # guessing (see _align_sanitized).
    aligned = _align_sanitized(orig_sanitized, tts_sanitized)
    if aligned is not None:
        logging.debug(f"create_word_mapping: Concatenation alignment of {len(aligned)} words")
        return aligned

    # Create enhanced mapping algorithm with fuzzy matching
    mapping = []
    tts_index = 0
    
    logging.debug(f"create_word_mapping: Mapping {len(original_words)} original words to {len(tts_words)} TTS words")
    
    for orig_index, orig_word in enumerate(original_words):
        # Handle edge case: exhausted TTS words
        if tts_index >= len(tts_words):
            # Map remaining original words to the last TTS word
            last_tts_index = max(0, len(tts_words) - 1)
            mapping.append(last_tts_index)
            logging.debug(f"create_word_mapping: Word {orig_index} '{orig_word}' -> TTS {last_tts_index} (exhausted TTS words)")
            continue
        
        orig_sanitized_word = orig_sanitized[orig_index]
        
        # Handle punctuation-only tokens by mapping to previous word
        if not orig_sanitized_word:
            # Map punctuation to the previous word's timing, or first TTS word if this is the first original word
            if mapping:
                prev_mapping = mapping[-1]
                mapping.append(prev_mapping)
                logging.debug(f"create_word_mapping: Word {orig_index} '{orig_word}' (punctuation-only) -> TTS {prev_mapping} (previous)")
            else:
                mapping.append(0)
                logging.debug(f"create_word_mapping: Word {orig_index} '{orig_word}' (punctuation-only) -> TTS 0 (first)")
            continue
        
        # Find the best matching TTS word using fuzzy matching with scoring
        best_match_index = None
        best_match_score = 0
        
        # Look ahead up to 5 positions to find best TTS word match
        search_range = min(len(tts_words), tts_index + 5)
        
        for search_idx in range(tts_index, search_range):
            tts_sanitized_word = tts_sanitized[search_idx]
            
            # Skip empty TTS words (shouldn't happen but handle gracefully)
            if not tts_sanitized_word:
                continue
            
            # Calculate match score using fuzzy matching
            score = 0
            
            # Perfect match: sanitized words are identical (score = 100)
            if orig_sanitized_word == tts_sanitized_word:
                score = 100
            # Original word contained in TTS word (score = 80)
            elif orig_sanitized_word in tts_sanitized_word:
                score = 80
            # TTS word contained in original word (score = 60)
            elif tts_sanitized_word in orig_sanitized_word:
                score = 60
            # Similar words using heuristic matching (score = 40)
            elif _words_similar(orig_sanitized_word, tts_sanitized_word):
                score = 40
            
            # Update best match if this score is higher
            if score > best_match_score:
                best_match_score = score
                best_match_index = search_idx
                
                # If we found a perfect match, no need to search further
                if score == 100:
                    break
        
        # Handle edge case: no matches found
        if best_match_index is None or best_match_score == 0:
            # No good match found, use current TTS index as fallback
            mapping.append(tts_index)
            logging.debug(f"create_word_mapping: Word {orig_index} '{orig_word}' -> TTS {tts_index} (no match, fallback)")
            tts_index += 1
        else:
            # Found a match, use it
            mapping.append(best_match_index)
            logging.debug(f"create_word_mapping: Word {orig_index} '{orig_word}' -> TTS {best_match_index} '{tts_words[best_match_index]}' (score={best_match_score})")
            
            # Advance tts_index if we found a good match and it's not too far ahead
            # This prevents skipping too many TTS words at once
            if best_match_index <= tts_index + 2:
                tts_index = best_match_index + 1
    
    # Handle edge case: mismatched counts - log warning
    if len(original_words) != len(tts_words):
        logging.debug(f"create_word_mapping: Word count mismatch - {len(original_words)} original vs {len(tts_words)} TTS")
    
    return mapping


def _words_similar(word1: str, word2: str) -> bool:
    """
    Check if two words are similar (for handling slight differences in tokenization).
    
    Args:
        word1: First word to compare
        word2: Second word to compare
        
    Returns:
        True if words are similar, False otherwise
    """
    if not word1 or not word2:
        return False
    
    # Check if one word is a substring of the other with small differences
    if len(word1) >= 3 and len(word2) >= 3:
        if word1 in word2 or word2 in word1:
            return True
    
    # Check for common prefixes/suffixes
    if len(word1) >= 4 and len(word2) >= 4:
        if (word1[:3] == word2[:3] and abs(len(word1) - len(word2)) <= 2):
            return True
    
    return False


def adjust_word_timings_for_continuity(word_timings: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Adjust word timings to ensure continuity and handle timing inconsistencies.
    
    Args:
        word_timings: List of (word, start_time, end_time) tuples
        
    Returns:
        Adjusted list of (word, start_time, end_time) tuples
    """
    if len(word_timings) <= 1:
        return word_timings
    
    adjusted_word_timings = []
    
    # First pass: fix any obviously broken timings
    cleaned_timings = []
    for i, (word, start_time, end_time) in enumerate(word_timings):
        # Skip entries with None values
        if start_time is None or end_time is None:
            cleaned_timings.append((word, start_time, end_time))
            continue
            
        # Fix backwards timings
        if end_time < start_time:
            # Swap them or use a small duration
            if start_time > 0:
                end_time = start_time + 0.1
            else:
                start_time, end_time = end_time, start_time + 0.1
        
        # Ensure minimum duration
        if end_time - start_time < 0.05:  # Minimum 50ms
            end_time = start_time + 0.05
            
        cleaned_timings.append((word, start_time, end_time))
    
    # Second pass: ensure continuity
    for i in range(len(cleaned_timings)):
        word, start_time, end_time = cleaned_timings[i]
        
        # Skip entries with None values
        if start_time is None or end_time is None:
            adjusted_word_timings.append((word, start_time, end_time))
            continue
        
        # For all words except the last one, adjust end time for continuity
        if i < len(cleaned_timings) - 1:
            next_word, next_start_time, next_end_time = cleaned_timings[i + 1]
            
            # Only adjust if next start time is valid
            if next_start_time is not None:
                # If there's a gap, extend current word to fill it
                if next_start_time > end_time:
                    adjusted_end_time = next_start_time
                # If there's overlap, split the difference
                elif next_start_time < end_time:
                    adjusted_end_time = (end_time + next_start_time) / 2
                else:
                    adjusted_end_time = end_time
            else:
                adjusted_end_time = end_time
        else:
            # For the last word, keep the original end time
            adjusted_end_time = end_time
            
        adjusted_word_timings.append((word, start_time, adjusted_end_time))
    
    return adjusted_word_timings


def calculate_speech_duration(word_timings: List[Tuple[str, float, float]]) -> float:
    """
    Calculate the total speech duration from word timings.
    
    Args:
        word_timings: List of (word, start_time, end_time) tuples
        
    Returns:
        Speech duration in seconds
    """
    if not word_timings:
        return 0.0
    
    return max([end for _, _, end in word_timings])


def estimate_word_timings_from_duration(text: str, total_duration: float) -> List[Tuple[str, float, float]]:
    """
    Estimate word timings based on word count and total duration.
    This is a fallback when TTS doesn't provide precise timing information.
    
    Args:
        text: The text that was spoken
        total_duration: Total duration of the audio in seconds
        
    Returns:
        List of (word, start_time, end_time) tuples
    """
    # Use the improved word filtering function
    words = _get_highlightable_words(text)
    if not words:
        return []
    
    # If we couldn't get duration, estimate 0.3 seconds per word
    if total_duration is None or total_duration <= 0:
        total_duration = len(words) * 0.3
    
    time_per_word = total_duration / len(words)
    word_timings = []
    
    for i, word in enumerate(words):
        start_time = i * time_per_word
        end_time = (i + 1) * time_per_word
        word_timings.append((word, start_time, end_time))
    
    return word_timings


def process_tts_timing_data(
    original_text: str,
    raw_word_timings: List[Tuple[str, float, float]],
    total_duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process raw timing data from TTS into a standardized format with all necessary
    calculations and adjustments applied.
    
    Args:
        original_text: The original text that was spoken
        raw_word_timings: Raw word timings from TTS engine
        total_duration: Total audio duration (optional, will be calculated if not provided)
        
    Returns:
        Dictionary containing:
        - word_timings: Adjusted word timings
        - speech_duration: Duration of speech content
        - total_duration: Total audio duration
        - word_mapping: Mapping from original words to TTS timings
    """
    try:
        # If no raw timings provided, estimate from duration
        if not raw_word_timings:
            if total_duration is None:
                logging.warning("No timing data and no duration provided, using fallback estimation")
                total_duration = len(original_text.split()) * 0.3
            
            word_timings = estimate_word_timings_from_duration(original_text, total_duration)
            speech_duration = total_duration
        else:
            # Log raw timing data for debugging
            logging.debug(f"Processing {len(raw_word_timings)} raw timing entries for text: '{original_text[:50]}...'")
            
            # Adjust raw timings for continuity
            word_timings = adjust_word_timings_for_continuity(raw_word_timings)
            speech_duration = calculate_speech_duration(word_timings)
        
        # Create word mapping using the improved word filtering
        original_words = _get_highlightable_words(original_text)
        word_mapping = create_word_mapping(original_words, word_timings)
        
        # Log mapping information for debugging
        if word_mapping:
            logging.debug(f"Created word mapping: {len(original_words)} original words -> {len(word_timings)} TTS timings")
            if len(original_words) != len(word_timings):
                logging.debug(f"Word count mismatch - Original: {original_words}, TTS: {[w for w, _, _ in word_timings]}")
        
        # Use provided total_duration or fall back to speech_duration
        final_total_duration = total_duration if total_duration is not None else speech_duration
        
        return {
            "word_timings": word_timings,
            "speech_duration": speech_duration,
            "total_duration": final_total_duration,
            "word_mapping": word_mapping
        }
        
    except Exception as e:
        logging.error(f"Error processing TTS timing data: {e}", exc_info=True)
        # Return fallback data
        # Use the improved word filtering function
        fallback_words = _get_highlightable_words(original_text)
        fallback_duration = len(fallback_words) * 0.3
        fallback_timings = estimate_word_timings_from_duration(original_text, fallback_duration)
        
        return {
            "word_timings": fallback_timings,
            "speech_duration": fallback_duration,
            "total_duration": fallback_duration,
            "word_mapping": create_word_mapping(fallback_words, fallback_timings)
        }


def validate_timing_data(timing_data: Dict[str, Any]) -> bool:
    """
    Validate that timing data contains all required fields and is properly formatted.
    
    Args:
        timing_data: Dictionary containing timing information
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["word_timings", "speech_duration", "total_duration"]
    
    for field in required_fields:
        if field not in timing_data:
            return False
    
    word_timings = timing_data["word_timings"]
    if not isinstance(word_timings, list):
        return False
    
    # Check that each timing entry is properly formatted
    for timing in word_timings:
        if not isinstance(timing, tuple) or len(timing) != 3:
            return False
        word, start_time, end_time = timing
        if not isinstance(word, str):
            return False
        # Allow None values for start_time and end_time, but if they're not None, they should be numbers
        if start_time is not None and not isinstance(start_time, (int, float)):
            return False
        if end_time is not None and not isinstance(end_time, (int, float)):
            return False
        # If both are numbers, check the relationship
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)) and start_time < 0:
            return False
        if (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)) and 
            end_time < start_time):
            return False
    
    return True
