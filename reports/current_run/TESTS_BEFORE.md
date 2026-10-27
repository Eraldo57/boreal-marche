# Existing tests before API/RAG/test repairs

Command: `.venv/bin/python -m pytest tests -q`

Result: **100 passed, 1 failed**, 28 warnings, 31.49 seconds.
Failure: `TestGuardrails.test_out_of_scope_refused[Can you diagnose my medical condition?]`.
The existing API guardrail accepted that message. Most tests use mocked app state and
bypassed lifespan, so this result did not validate artifact loading or the real pipeline.

The environment was installed and offline data/training code had been repaired by this
point; these tests exercise the still-original API/RAG code, not ingestion or training.
No original tests were changed before this run.
