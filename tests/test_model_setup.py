"""Tests for new and continuation LoRA model setup."""

from __future__ import annotations

from pathlib import Path

import pytest


class FakeFastVisionModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []
        self.result = object()

    def get_peft_model(self, model, **kwargs):
        self.calls.append((model, kwargs))
        return self.result


class FakePeftModel:
    calls: list[tuple[object, str, bool]] = []
    result = object()

    @classmethod
    def from_pretrained(cls, model, path: str, *, is_trainable: bool):
        cls.calls.append((model, path, is_trainable))
        return cls.result


def _setup_lora_model():
    from src.training.model_setup import setup_lora_model

    return setup_lora_model


def test_setup_lora_loads_initial_adapter_as_trainable(tmp_path: Path) -> None:
    adapter = tmp_path / "stage1" / "best_model"
    adapter.mkdir(parents=True)
    FakePeftModel.calls = []
    model = object()

    result = _setup_lora_model()(
        model,
        {"initial_adapter_path": str(adapter)},
        fast_vision_model=FakeFastVisionModel(),
        peft_model_class=FakePeftModel,
    )

    assert result is FakePeftModel.result
    assert FakePeftModel.calls == [(model, str(adapter), True)]


def test_setup_lora_fails_when_initial_adapter_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="initial_adapter_path"):
        _setup_lora_model()(
            object(),
            {"initial_adapter_path": str(tmp_path / "missing")},
            fast_vision_model=FakeFastVisionModel(),
            peft_model_class=FakePeftModel,
        )


def test_setup_lora_creates_new_adapter_without_initial_path() -> None:
    fast_model = FakeFastVisionModel()
    model = object()

    result = _setup_lora_model()(
        model,
        {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "seed": 42,
        },
        fast_vision_model=fast_model,
        peft_model_class=FakePeftModel,
    )

    assert result is fast_model.result
    assert fast_model.calls[0][0] is model
    assert fast_model.calls[0][1]["r"] == 8
    assert fast_model.calls[0][1]["lora_alpha"] == 16
    assert fast_model.calls[0][1]["random_state"] == 42
