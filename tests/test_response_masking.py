"""Tests for response-only supervision validation."""

from __future__ import annotations

import importlib

import pytest


class _FakeTokenizer:
    def __init__(self, decoded: str) -> None:
        self.decoded = decoded

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert token_ids
        assert skip_special_tokens is True
        return self.decoded


def _validator():
    module = importlib.import_module("src.training.response_masking")
    return module.validate_response_only_batch


def test_validate_response_only_batch_accepts_exact_assistant_target() -> None:
    """A correctly masked batch should expose only the expected answer."""
    stats = _validator()(
        {
            "input_ids": [[10, 11, 12, 13]],
            "labels": [[-100, -100, 12, 13]],
        },
        _FakeTokenizer("C C O"),
        expected_response="CCO",
    )

    assert stats == {
        "sequence_tokens": 4,
        "supervised_tokens": 2,
        "decoded_response": "C C O",
    }


def test_validate_response_only_batch_rejects_fully_masked_labels() -> None:
    """A marker mismatch must stop before starting an expensive run."""
    with pytest.raises(RuntimeError, match="no supervised assistant tokens"):
        _validator()(
            {
                "input_ids": [[10, 11]],
                "labels": [[-100, -100]],
            },
            _FakeTokenizer(""),
            expected_response="CCO",
        )


def test_validate_response_only_batch_rejects_prompt_leakage() -> None:
    """Labels containing prompt text must not pass the startup check."""
    with pytest.raises(RuntimeError, match="does not match the expected target"):
        _validator()(
            {
                "input_ids": [[10, 11, 12]],
                "labels": [[10, 11, 12]],
            },
            _FakeTokenizer("Analyze the NMR. CCO"),
            expected_response="CCO",
        )
