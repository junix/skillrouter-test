"""Contract tests for ``_last_token_pool`` — last-non-pad-token pooling.

Contract (from models.py):
  - For each sequence, return the hidden state at the last position whose
    attention_mask is 1 (the last real token).
  - Left-padding fast path: if every row's final column is attended
    (mask[:, -1].sum() == batch), the last column IS the last real token for
    every row -> return last_hidden[:, -1] without per-row indexing.
  - Otherwise compute seq_lens = attention_mask.sum(dim=1) - 1 and gather.

Shape contract: output is (batch, hidden_dim).
"""

import pytest
import torch

from skillrouter_test.models import _last_token_pool


def _seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


# --- right-padding (general) path -----------------------------------------

def test_right_padding_picks_last_attended_token_per_row() -> None:
    # Row 0 fully attended (len 3) -> idx 2; row 1 len 2 -> idx 1.
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
         [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]
    )
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = _last_token_pool(hidden, mask)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor([[7.0, 8.0, 9.0], [13.0, 14.0, 15.0]]))


def test_all_padding_row_yields_index_minus_one() -> None:
    # seq_lens for an all-zero mask row is -1; the contract does not guard this
    # (it is an invalid input for the encoder), but we pin the actual behavior so
    # a future guard surfaces as a test change rather than silent drift.
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = torch.tensor([[0, 0]])
    out = _last_token_pool(hidden, mask)
    # index -1 -> last column
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))


def test_shape_preserves_hidden_dim() -> None:
    hidden = torch.randn(4, 7, 16)
    mask = torch.ones(4, 7, dtype=torch.long)
    out = _last_token_pool(hidden, mask)
    assert out.shape == (4, 16)


def test_single_token_sequence() -> None:
    hidden = torch.tensor([[[5.0, 6.0]]])
    mask = torch.tensor([[1]])
    out = _last_token_pool(hidden, mask)
    assert torch.allclose(out, torch.tensor([[5.0, 6.0]]))


# --- left-padding fast path ------------------------------------------------

def test_left_padding_fast_path_returns_last_column() -> None:
    # Both rows left-padded: the only attended tail token is the last column.
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
         [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]
    )
    mask = torch.tensor([[0, 1, 1], [0, 1, 1]])  # mask[:,-1].sum() == batch -> fast path
    out = _last_token_pool(hidden, mask)
    assert torch.allclose(out, torch.tensor([[7.0, 8.0, 9.0], [16.0, 17.0, 18.0]]))


def test_fast_path_does_not_trigger_when_any_last_mask_is_zero() -> None:
    # One row has last mask == 0 -> fast path NOT taken -> per-row gather via
    # seq_lens = sum(dim=1) - 1 (the general-path last-attended index for this
    # convention).
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
         [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]
    )
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])  # row 1 last col unattended
    out = _last_token_pool(hidden, mask)
    # row 0: seq_len 2 -> idx 1 (4,5,6); row 1: seq_len 2 -> idx 1 (13,14,15)
    assert torch.allclose(out, torch.tensor([[4.0, 5.0, 6.0], [13.0, 14.0, 15.0]]))


# --- properties within each padding convention -----------------------------
#
# The two code paths serve DIFFERENT padding conventions and are NOT equivalent
# on arbitrary masks:
#   * general path uses seq_lens = sum(dim=1) - 1, which is the last-attended
#     index only when padding is RIGHT-aligned (trailing zeros).
#   * fast path returns the last column, which is the last real token only when
#     padding is LEFT-aligned (the tokenizer's padding_side="left") and the last
#     column is attended.
# So we assert each path against its OWN convention, never cross-convention.


def test_general_path_last_attended_index_matches_for_right_padding() -> None:
    # Property: for right-padded batches, every output row equals the hidden
    # state at its last attended position (sum-1).
    _seed(7)
    hidden = torch.randn(6, 10, 4)
    masks = []
    for _ in range(6):
        n = torch.randint(1, 11, ()).item()  # at least one attended token
        masks.append([1] * n + [0] * (10 - n))  # right padding
    mask = torch.tensor(masks)
    out = _last_token_pool(hidden, mask)
    manual = torch.stack([hidden[i, mask[i].sum() - 1] for i in range(6)])
    assert torch.allclose(out, manual)


def test_fast_path_returns_last_column_for_left_padding() -> None:
    # Property: for left-padded batches whose last column is attended, the fast
    # path returns exactly the last column for every row.
    _seed(3)
    hidden = torch.randn(5, 12, 8)
    masks = []
    for _ in range(5):
        n = torch.randint(0, 12, ()).item()
        masks.append([0] * n + [1] * (12 - n))  # left padding; last col always 1
    mask = torch.tensor(masks)
    out = _last_token_pool(hidden, mask)
    assert torch.allclose(out, hidden[:, -1, :])
