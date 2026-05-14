# Progress

## Status
Completed

## Tasks
- Compared FinBERT variants, CardiffNLP/Twitter RoBERTa, VADER, TextBlob, and LLM/API options.
- Recommended MVP approach for a small local Python university project.
- Wrote actionable implementation shape for a swappable sentiment analyzer.

## Files Changed
- `/Users/diego/UNED/Aplicaciones/tarea_final/progress.md`
- `/Users/diego/UNED/Aplicaciones/tarea_final/research.md`

## Notes
- Recommendation: implement VADER first as deterministic local/offline MVP fallback, then add optional FinBERT backend for finance-aware quality.
- Normal tests should not depend on FinBERT downloads; use deterministic analyzer tests and optional skipped ML integration tests.
- Web-search tooling was not available in this subagent environment, so the brief cites known primary docs/model pages but could not refresh them live.
