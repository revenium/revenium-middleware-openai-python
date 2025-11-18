# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.7] - 2025-11-14

## [0.4.7] - 2025-11-13

### Fixed
- LangChain integration with langchain-openai now works correctly (handles LegacyAPIResponse)
- Removed invalid use_async parameter from langchain example
- Added langchain-openai to [langchain] optional dependency
- Streaming metering edge case where empty chunks could skip final usage tracking
- Documentation metadata examples now use correct nested subscriber structure

### Added
- examples/getting_started.py - simple entry point example
- Azure credentials note in README
- LangChain installation instructions in README
- CHANGELOG.md for version history tracking
- LangChain integration with langchain-openai
- Streaming metering for edge cases
- Documentation metadata examples
- Getting started example
- LangChain installation instructions

## [0.4.6] - 2024-06-18

### Changed
- Updated revenium_middleware dependency to 0.3.4

## [0.4.5] - 2024-06-17

### Changed
- Updated license from Apache 2.0 to MIT
- Updated revenium_middleware dependency

## [0.4.3] - 2024-06-17

### Added
- Middleware source tracking

## [0.4.2] - 2024-06-17

### Added
- LangChain integration with unified handler architecture

## [0.4.1] - 2024-06-17

### Changed
- Updated revenium_middleware dependency to 0.2.9

## [0.3.9] - 2024-06-17

### Changed
- Updated README to clarify nested subscriber object structure

## [0.3.8] - 2024-06-16

### Added
- Support for nested subscriber metadata structure with credential object

### Fixed
- Maintained nested credential structure in subscriber object

## [0.3.7] - 2024-05-10

### Changed
- Updated dependencies

## [0.3.6] - 2024-05-04

### Changed
- Updated middleware dependencies

## [0.3.4] - 2024-04-21

### Changed
- Updated OpenAI resource path in middleware

## [0.3.3] - 2024-04-20

### Changed
- Updated dependencies

## [0.3.2] - 2024-04-19

### Changed
- Removed documentation section from README

## [0.3.1] - 2024-04-18

### Added
- Subscriber credential name to usage metadata

## [0.2.8] - 2024-04-10

### Added
- Subscriber email and credential to usage logging

### Changed
- Removed ai_provider_key_name from metadata

## [0.2.6] - 2024-04-02

### Changed
- Removed source_id from README

## [0.2.5] - 2024-03-30

### Added
- Model source detection based on system fingerprint

### Changed
- Renamed model_source to provider for clarity

### Fixed
- Correctly log system fingerprint in completion usage
- Handle missing cached_tokens in OpenAI response

## [0.2.4] - 2024-03-25

### Added
- Logging section to README with usage instructions
- Updated logging to use logger instance

## [0.2.3] - 2024-03-23

### Changed
- Updated dependencies

## [0.2.2] - 2024-03-22

### Added
- Streaming support with accurate token counting
- Time to first token metric tracking

### Fixed
- Token counting for streaming OpenAI responses

## [0.2.1] - 2024-03-22

### Changed
- Updated license classifier to Apache Software License
- Added Apache License 2.0

## [0.2.0] - 2024-03-22

### Changed
- Updated dependencies
- Replaced metering module with core functionality

### Fixed
- Resolved async metering call issue

## [0.1.0] - 2024-03-16

### Added
- Initial release
- OpenAI chat completion metering
- Embedding support
- Azure OpenAI support
- Usage metadata tracking
- Comprehensive test coverage

[0.4.7]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.4.3...v0.4.5
[0.4.3]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.9...v0.4.1
[0.3.9]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.4...v0.3.6
[0.3.4]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.8...v0.3.1
[0.2.8]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.6...v0.2.8
[0.2.6]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/revenium/revenium-middleware-openai-python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/revenium/revenium-middleware-openai-python/releases/tag/v0.1.0
