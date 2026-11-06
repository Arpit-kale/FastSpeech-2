"""FastSpeech 2 for English and Telugu.

Adapted from ming024/FastSpeech2 (MIT). The main changes are documented in
``docs/MODIFICATIONS.md``: learned (MFA-free) alignment, a rule-based Telugu
G2P, language-aware text normalisation, speaker-wise pitch statistics,
frame-budget batching and a self-contained inference checkpoint.
"""

__version__ = "1.0.0"
