"""Q2 codebase (filters + resampling), self-contained copy.

Source: ``JPM_MLCOE/Q2/Q2/src/{filters,resampling}/``.

Used unchanged except for the cumulative-weight bug fix in
``filters/pf.py`` (see report appendix B / ``implementation.md`` Section 3.4).
The fix is marked with ``# CUMULATIVE FIX`` comments at the affected
lines so the original is recoverable in one diff.

Wrapped to our dynamic Deep-Halo SSM via ``inference.custom.ssm_wrapper``.
"""
