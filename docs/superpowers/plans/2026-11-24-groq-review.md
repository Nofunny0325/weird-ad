# Groq-assisted food advertisement review

The user approved combining Groq speech recognition and context review with the existing local evidence checks. The target is improved detection and lower host resource use within the free API tier.

## Design

Use Whisper Large v3 for timestamped Korean transcription and GPT-OSS 120B with strict structured output for context review. Keep every extracted segment eligible for review, including segments without dictionary hits. Give the model the registered rules, product record when available, and adjacent timed text. Constrain its choices to supplied clause IDs and rule IDs. Reconstruct the full source clause on the server and check caution and denial context independently. Attach source links and timestamps from local records. Cloud findings remain review candidates.

Use a server-side GROQ_API_KEY and a per-submission Groq checkbox. Show the external processing scope on both forms. Cache no uploaded content or model results. Keep timeout, response-size and request-size bounds. A rate-limit cooldown stops repeated calls after 429. Local review remains available when a key is missing, the user chooses local processing, or the provider fails. Report partial cloud coverage.

The current Render free deployment, upload limits, local models and product-specific comparisons remain supported. Measure real provider performance after authentication, and distinguish synthetic functional checks from real-ad accuracy evaluation.

## Implementation plan

- [x] Add a bounded Groq HTTP client, timestamped transcription validation and rate-limit handling. Test HTTP failures, cooldowns and malformed responses.
- [x] Add context batches that cover all extracted text with overlap. Validate strict output, exact quotes and registered rule IDs before creating timed findings. Test no-keyword statements, prompt injection, ungrounded output and partial coverage.
- [x] Connect Flask and CLI to opt-in hybrid review, local fallback and accurate processing notices. Test keyless operation and real local video entry points.
- [x] Run regression tests and synthetic provider checks, push the verified code, monitor Render deployment and verify public routes.
- [ ] Connect a user-authorized Groq key, verify live transcription and context results, and report measured latency and evaluation limits.

## Verification and account connection

37 automated tests passed, including actual local E5 inference. A synthetic Korean video passed the local OCR and isolated Whisper path and the mocked Groq upload-to-render path. The final review found and fixed truncated HTTP responses, out-of-audio timestamps and a negated clause shortened into a positive-looking quote. Live checks also exposed cross-disease evidence leakage. A general disease/medicine rule based on the MFDS public reporting guide now handles claims outside the scope of a disease-specific criterion.

The user explicitly approved accepting the Groq agreement, creating a service key and storing it in Render. GitHub authentication was restored and the service key was registered on 2026-11-27. A live synthetic Korean video completed Groq transcription and context review in 31.86 seconds, including server-side OCR and E5. A more complex synthetic text exposed output truncation and unstable verbatim copying. The output budget was increased and clause/rule IDs were constrained at decoding time. The updated mixed-case comparison still needs a final live check. Real-ad detection accuracy remains unmeasured.
