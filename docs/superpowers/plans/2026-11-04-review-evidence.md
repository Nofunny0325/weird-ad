# Evidence-based screening implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace keyword convictions with evidence-preserving, uncertainty-aware screening.
**Architecture:** Retain Flask, OCR/STT and the existing matcher. Carry extraction warnings and timed matches into a conservative judge result. Strictly validate optional AI responses and display limits in both web and CLI results.
**Tech Stack:** Python, Flask, unittest; existing OCR/STT adapters.
**Spec:** ../../PROJECT_STATUS.md and the user-approved first-step proposal in the conversation.

## Global constraints

No automatic product-registry lookup or clinical validation in this increment. No paid API calls in verification. Preserve the existing repository architecture. Keywords alone never establish illegality. Missing evidence never establishes safety.

## Review focus

Empty text, skipped/failed STT, negated/quoted keywords, malformed AI JSON including string booleans, and duplicate OCR/STT statements must retain uncertainty and source locations.

## Tasks

- [x] Add regression tests for empty/no-match results, keyword-only uncertainty, AI failures and strict parsing, timed sources and partial extraction.
- [x] Run tests and confirm the old behavior fails assertions.
- [x] Extend extraction result with warnings and channel status; preserve partial results on extractor failure.
- [x] Extend matches/results with times, sources, reasons and limits; analyze both keyword lists in context; treat errors as review required.
- [x] Update Flask template and CLI documentation; show missing checks and evidence without legal certification.
- [x] Run the complete test suite and web integration checks; inspect diff and commit to the user's fork branch.

## Verification ledger

Tests were observed failing before implementation. A fresh reviewer found cross-segment phrases and OCR-displaced speech negation; both received failing regressions and fixes. OCR repeated captions now extend estimated intervals and reappear after blank frames. Actual ffmpeg, Tesseract and Whisper engines and a live API key were unavailable; no real-media or paid API validation claimed. Test evidence uses deterministic adapters. New local configuration defaults to loopback and debug off. Product/clinical checks remain pending and prevent automatic clearance.
