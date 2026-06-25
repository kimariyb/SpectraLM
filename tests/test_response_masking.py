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


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return "rendered prompt"


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


def test_validate_response_only_batch_rejects_thinking_prefix() -> None:
    """No thinking tags should be supervised in text-only non-thinking SFT."""
    with pytest.raises(RuntimeError, match="thinking tokens"):
        _validator()(
            {
                "input_ids": [[10, 11, 12, 13]],
                "labels": [[-100, -100, 12, 13]],
            },
            _FakeTokenizer("<think>\n\n</think>\n\nCCO"),
            expected_response="CCO",
        )


def test_non_thinking_generation_prompt_disables_qwen_thinking() -> None:
    """Inference must begin after Qwen's empty non-thinking prefix."""
    module = importlib.import_module("src.training.response_masking")
    processor = _FakeProcessor()
    messages = [{"role": "user", "content": "Predict."}]

    rendered = module.apply_non_thinking_chat_template(processor, messages)

    assert rendered == "rendered prompt"
    assert processor.calls == [
        (
            messages,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
    ]


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
