# HERON Development TODO

This document tracks planned improvements and enhancements for the HERON framework.

## Infrastructure

### CI/CD
- [x] Add GitHub Actions workflow for CI/CD (tests, linting, type checking)
- [x] Configure Sphinx autodoc for API reference generation
- [x] Update API documentation for both `heron` and `powergrid` modules

### Future Work
- [ ] Add RL policy wrapper utilities for common RL frameworks (Stable-Baselines3, CleanRL)
- [ ] Implement Kafka/Redis message brokers for production distributed deployments
- [ ] Add Docker/Kubernetes deployment examples
- [ ] Create pre-configured setup data files with load profiles

## Documentation

- [x] API reference for `heron` core modules
- [x] API reference for `powergrid` case study modules
- [ ] Add troubleshooting guide
- [ ] Add performance benchmarks and guidelines
- [ ] Add production deployment guide

## Code Quality

- [ ] Populate `heron/__init__.py` with public API exports
- [ ] Add comprehensive docstrings to all public APIs
- [ ] Increase test coverage for heron core modules
