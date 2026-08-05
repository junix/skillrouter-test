"""Test harness for the SkillRouter retrieve-and-rerank pipeline.

Paper: SkillRouter: Skill Routing for LLM Agents at Scale (arXiv:2603.22455).
Models:
  - encoder  : pipizhao/SkillRouter-Embedding-0.6B  (SR-Emb-0.6B)
  - reranker : pipizhao/SkillRouter-Reranker-0.6B    (SR-Rank-0.6B)
"""

# Lazy: importing the package (e.g. for `skillrouter available` / `doctor`,
# which must stay fast and not load the model stack) must NOT pull in
# torch/transformers. The model classes are imported on first attribute access.
__all__ = ["Skill", "SkillEncoder", "SkillReranker", "pick_device"]


def __getattr__(name):
    if name in __all__:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module 'skillrouter_test' has no attribute {name!r}")
