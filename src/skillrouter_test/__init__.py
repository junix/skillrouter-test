"""Test harness for the SkillRouter retrieve-and-rerank pipeline.

Paper: SkillRouter: Skill Routing for LLM Agents at Scale (arXiv:2603.22455).
Models:
  - encoder  : pipizhao/SkillRouter-Embedding-0.6B  (SR-Emb-0.6B)
  - reranker : pipizhao/SkillRouter-Reranker-0.6B    (SR-Rank-0.6B)
"""

from .models import Skill, SkillEncoder, SkillReranker, pick_device

__all__ = ["Skill", "SkillEncoder", "SkillReranker", "pick_device"]
