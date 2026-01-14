# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Add ScatterNode for parallel workflow execution (#2)

### Fixed

### Changed
- Fix PR #2 review feedback for steady-clock polling (#12)
- Add tests for ScatterNode parallel execution (#6)
- Add gather mechanism to collect results from parallel instances (#5)
- Implement parallel workflow execution with asyncio (#4)
- Define ScatterNode base class with State and scatter() interface (#3)
