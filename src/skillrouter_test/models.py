"""Thin wrappers around the two SkillRouter checkpoints.

The loading / pooling / prompt conventions follow the model cards on
Hugging Face. Local checkpoints can be substituted via the environment
variables ``SR_EMB_PATH`` and ``SR_RANK_PATH`` (mirrors the upstream repo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

EMB_MODEL_ID = os.environ.get("SR_EMB_PATH", "pipizhao/SkillRouter-Embedding-0.6B")
RANK_MODEL_ID = os.environ.get("SR_RANK_PATH", "pipizhao/SkillRouter-Reranker-0.6B")

QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant skill "
    "document that would help an agent complete the task\nQuery:"
)
RERANK_INSTRUCTION = (
    "Given a task description, judge whether the skill document is relevant "
    "and useful for completing the task"
)
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Answer only "yes" or "no".'
)


@dataclass
class Skill:
    """One routable skill: name, short description, full implementation body."""

    name: str
    description: str = ""
    body: str = ""

    def as_document(self, desc_max: int = 500, body_max: int = 2000) -> str:
        """Flatten to the ``name | description | body`` format the models expect."""
        return f"{self.name} | {self.description[:desc_max]} | {self.body[:body_max]}"


def pick_device(requested: str | None = None) -> str:
    """Resolve a torch device string, preferring cuda > mps > cpu."""
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str) -> torch.dtype:
    return torch.bfloat16 if device == "cuda" else torch.float32


def _last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-pad token (handles left padding)."""
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_lens]


class SkillEncoder:
    """First stage: bi-encoder retrieval over the full skill text."""

    def __init__(self, model_id: str = EMB_MODEL_ID, device: str | None = None, max_length: int = 4096) -> None:
        self.device = pick_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=_dtype_for(self.device))
            .eval()
            .to(self.device)
        )

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of raw texts into L2-normalized embeddings."""
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        embs = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(embs, p=2, dim=1)

    def encode_query(self, query: str) -> torch.Tensor:
        return self.encode([f"{QUERY_INSTRUCTION}{query}"])

    def encode_skills(self, skills: list[Skill]) -> torch.Tensor:
        return self.encode([s.as_document() for s in skills])

    def rank(self, query: str, skills: list[Skill]) -> list[tuple[int, float]]:
        """Return (skill_index, cosine_score) sorted best-first."""
        q = self.encode_query(query)
        docs = self.encode_skills(skills)
        scores = (q @ docs.T).squeeze(0)
        order = torch.argsort(scores, descending=True).tolist()
        return [(i, float(scores[i])) for i in order]


class SkillReranker:
    """Second stage: cross-encoder reranking of a small candidate set."""

    def __init__(self, model_id: str = RANK_MODEL_ID, device: str | None = None) -> None:
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_dtype_for(self.device)).eval().to(self.device)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")

    def _prompt(self, query: str, skill: Skill) -> str:
        body = (
            f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {skill.as_document()}"
        )
        return (
            f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n"
            f"<|im_start|>assistant\n<think></think>"
        )

    @torch.no_grad()
    def score(self, query: str, skill: Skill) -> float:
        """Relevance score = logit(yes) - logit(no) at the final position."""
        enc = self.tokenizer(self._prompt(query, skill), return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits[:, -1, :]
        return float(logits[0, self.yes_id] - logits[0, self.no_id])

    def rerank(self, query: str, skills: list[Skill]) -> list[tuple[int, float]]:
        """Return (skill_index, score) sorted best-first."""
        scored = [(i, self.score(query, s)) for i, s in enumerate(skills)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored
