# Delta for UI Highlighting

## MODIFIED Requirements

### Requirement: Highlight Mode Auto-Adaptation
The system MUST automatically adjust highlighting behavior when a TTS provider is selected that does not support word-level timing.

#### Scenario: Switch to OpenAI TTS
- Given: Word highlighting is currently enabled (WORD_HIGHLIGHT_MODE > 0)
- When: The user switches to a TTS provider with supports_word_timing = False
- Then: WORD_HIGHLIGHT_MODE is automatically set to 0 and sentence-level highlighting is used

#### Scenario: Switch to Edge TTS
- Given: Word highlighting was previously disabled due to a non-timing provider
- When: The user switches to Edge TTS (supports_word_timing = True)
- Then: The previous word highlighting mode is restored

#### Scenario: User Override While Using Non-Timing Provider
- Given: WORD_HIGHLIGHT_MODE was auto-set to 0 for a non-timing provider
- When: The user manually sets WORD_HIGHLIGHT_MODE to 1 or 2
- Then: WORD_HIGHLIGHT_MODE is set to the user's chosen value
- And: The saved "previous mode" is NOT overwritten by this manual change

### Requirement: Previous Highlight Mode Preservation
The system MUST preserve the user's previous WORD_HIGHLIGHT_MODE setting before automatically disabling word highlighting for a non-timing provider, and restore it when a timing-capable provider is selected.

#### Scenario: Mode Saved Before Switch
- Given: WORD_HIGHLIGHT_MODE is set to 2 (standout mode)
- When: A non-timing TTS provider is selected
- Then: The value 2 is saved to internal storage
- And: WORD_HIGHLIGHT_MODE is set to 0

#### Scenario: Mode Restored After Switch Back
- Given: A previous WORD_HIGHLIGHT_MODE was saved from a prior timing-capable provider session
- When: A timing-capable TTS provider is selected
- Then: The saved WORD_HIGHLIGHT_MODE value is restored
- And: No stale saved mode affects the current selection
