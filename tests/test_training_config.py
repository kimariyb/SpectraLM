"""Tests for training configuration compatibility helpers."""

from __future__ import annotations

import argparse

from spectralm.training.train import (
    build_sft_config_kwargs,
    normalize_training_special_tokens,
    resolve_eos_token,
)


class FakeTokenizer:
    """Minimal tokenizer-like object for EOS resolution tests."""

    eos_token = "<|im_end|>"
    pad_token = None
    unk_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        """Return a small fake vocabulary.

        Returns
        -------
        dict[str, int]
            Fake token vocabulary.
        """
        return {"<|im_end|>": 1, "hello": 2}


class FakeProcessor:
    """Minimal processor-like object with a nested tokenizer."""

    tokenizer = FakeTokenizer()


class FakeSFTConfig:
    """Fake SFTConfig signature that supports EOS tokens."""

    def __init__(self, output_dir: str, eos_token: str | None = None, max_length: int | None = None) -> None:
        """Initialize fake config.

        Parameters
        ----------
        output_dir
            Output directory.
        eos_token
            EOS token value.
        max_length
            Maximum sequence length.
        """
        self.output_dir = output_dir
        self.eos_token = eos_token
        self.max_length = max_length


def test_resolve_eos_token_from_nested_processor_tokenizer() -> None:
    """EOS resolution should use the processor's nested tokenizer."""
    assert resolve_eos_token(FakeProcessor()) == "<|im_end|>"


def test_build_sft_config_kwargs_sets_valid_eos_token() -> None:
    """SFT kwargs should override TRL placeholder EOS defaults."""
    args = argparse.Namespace(
        output_dir="outputs/test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=1,
        eval_steps=1,
        seed=1,
        max_seq_length=128,
    )
    kwargs = build_sft_config_kwargs(args, FakeSFTConfig, FakeProcessor())
    assert kwargs["eos_token"] == "<|im_end|>"
    assert kwargs["max_length"] == 128


def test_normalize_training_special_tokens_replaces_trl_placeholder() -> None:
    """Post-construction normalization should replace TRL placeholder tokens."""
    training_args = argparse.Namespace(eos_token="<EOS_TOKEN>", pad_token="<PAD_TOKEN>")
    normalize_training_special_tokens(training_args, FakeProcessor())
    assert training_args.eos_token == "<|im_end|>"
    assert training_args.pad_token == "<|im_end|>"
