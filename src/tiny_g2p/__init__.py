"""tiny_g2p -- tiny parallel per-character Polish G2P.

Follows the experiments in vercel-labs/gpu-lexer (a tiny learned lexer with a
guarded training regime), applied to Polish grapheme-to-phoneme conversion:

* the model labels every character in parallel (phone or blank), like gpu-lexer
  labels every part in parallel -- no autoregression, ~15K parameters;
* per-character labels come from a monotonic DP aligner, the analogue of their
  Shiki teacher;
* training follows their regime: warm-start continuation, quantization-aware
  final epochs, failure-bank fine-tuning, a fixed verification split that is
  never trained on, and promotion only on strict guarded improvement.

Quick start::

    tiny-g2p fetch                 # pinned MFA dictionary (or reuse sibling's)
    tiny-g2p train --epochs 32     # warm-starts active/, promotes iff improved
    tiny-g2p eval                  # word accuracy + PER on held-out test words
    tiny-g2p predict "Wrocław"     # transcribe words with the promoted model
"""

__version__ = "0.1.0"
