"""iRDM -- improved Representation Distribution Matching for one-step visual generation.

A one-step image generator is trained with no teacher, adversary, or trajectory by
matching generated and real feature distributions under a battery of frozen pretrained
encoders. The library is organized around the paper's two design axes:

* :mod:`rdm.compare` -- the comparison axis: kernels, Nystrom math, and all distances
  (the iRDM loss plus the ablation distances).
* :mod:`rdm.representation` -- the representation axis: the encoder battery, the
  generators, and the joint image-text feature.

with :mod:`rdm.refprep` (the offline frozen reference), :mod:`rdm.train` (the loop,
gradient caching, and the proportional-Lagrangian controller), :mod:`rdm.eval` (the
off-objective metrics), and :mod:`rdm.toy` (the low-dimensional diagnostics).
"""
__version__ = "0.1.0"
