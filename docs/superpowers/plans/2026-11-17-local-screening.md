# Local ad screening

**Goal:** Review Korean advertising claims across all extracted text without sending uploads or transcripts to a hosted inference service.

**Design:** Keep Flask and the timed OCR/STT pipeline. Run multilingual E5 on the server CPU for candidate retrieval, with character n-gram TF-IDF as the no-model fallback. Use separate high-signal checks and exact product evidence for review decisions. User-visible evidence retains text, timestamps, source and rule.

**Constraints:** Preserve uncertainty when evidence is missing. Models are installed separately and inference is local-only. There is no automatic legal clearance. Do not treat editorial examples as a performance dataset.

- [x] Add tests for no-keyword claims, cautions, adjacent segments, and exact product evidence.
- [x] Connect the local word lists, context rules, reference catalog and local semantic retrieval.
- [x] Convert the upload page, CLI and setup to local inference, with text testing and temporary files.
- [x] Verify the real model, model-free behavior and selected end-to-end upload cases.
