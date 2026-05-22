# Delta for TTS Provider

## ADDED Requirements

### Requirement: OpenAI TTS Provider Implementation
The system SHALL provide an OpenAI TTS provider that generates audio from text using the OpenAI Speech API, following the existing `TTSBase` plugin interface convention.

#### Scenario: Provider Initialization
- Given: The `openai` Python package is installed and `OPENAI_API_KEY` environment variable is set
- When: The OpenAI TTS provider is initialized
- Then: An `AsyncOpenAI` client is created and `self.initialized` is set to `True`

#### Scenario: Missing API Key
- Given: The `OPENAI_API_KEY` environment variable is not set
- When: The OpenAI TTS provider attempts to initialize
- Then: Initialization fails with a clear error message instructing the user to set the environment variable

#### Scenario: Missing Package
- Given: The `openai` Python package is not installed
- When: The OpenAI TTS provider attempts to initialize
- Then: Initialization fails with a message instructing the user to install the package

#### Scenario: Audio Generation
- Given: The provider is initialized and valid text is provided
- When: `generate_audio(text, output_path)` is called
- Then: Audio is generated using the OpenAI Speech API and saved to the output path as an mp3 file

### Requirement: No Word-Level Timing Support
The OpenAI TTS provider SHALL indicate that it does not support word-level timing data, causing the system to use sentence-level highlighting instead.

#### Scenario: Word Timing Check
- Given: An initialized OpenAI TTS provider
- When: `supports_word_timing` property is accessed
- Then: It returns `False`

#### Scenario: Timing Data Request
- Given: An initialized OpenAI TTS provider
- When: `get_raw_timing_data(text, output_path)` is called
- Then: It returns an empty list `[]`

### Requirement: TTSBase Word Timing Capability Property
The `TTSBase` abstract class SHALL expose a `supports_word_timing` property that indicates whether a provider can return accurate word-level timing data.

#### Scenario: Default Word Timing Support
- Given: A TTS provider that extends `TTSBase` without overriding the property
- When: `supports_word_timing` is accessed
- Then: It returns `True` (backward compatible with existing providers)

#### Scenario: OpenAI Provider Override
- Given: The `OpenAITTS` provider class
- When: `supports_word_timing` is accessed
- Then: It returns `False`


### Requirement: OpenAI TTS Warm-Up Support
The OpenAI TTS provider SHALL implement a warm-up mechanism that reduces first-call latency after initialization.

#### Scenario: Warm-Up Reduces Latency
- Given: An initialized OpenAI TTS provider
- When: `warm_up()` is called before the first `generate_audio()` call
- Then: The first audio generation completes with reduced latency compared to a cold start

#### Scenario: Warm-Up Failure Is Non-Fatal
- Given: An initialized OpenAI TTS provider
- When: `warm_up()` is called and the API request fails
- Then: The error is logged as a warning
- And: The provider remains available for normal audio generation

### Requirement: OpenAI Default Voice Configuration
The configuration module SHALL include a default voice mapping for the OpenAI TTS provider.

#### Scenario: Default Voice Lookup
- Given: The `TTS_VOICES` configuration dictionary
- When: Looking up the voice for `"openai"`
- Then: It returns `"alloy"` as the default voice

### Requirement: OpenAI Optional Dependency
The project SHALL include OpenAI as an optional dependency that users can install separately.

#### Scenario: Install OpenAI Extra
- Given: A user wants to use OpenAI TTS
- When: They run `pip install lue[openai]`
- Then: The `openai>=1.0.0` package is installed
