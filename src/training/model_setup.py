"""Pure selection of new or continued LoRA adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def setup_lora_model(
    model: Any,
    config: dict[str, Any],
    *,
    fast_language_model: Any,
    peft_model_class: Any,
) -> Any:
    """Create a new adapter or continue an existing trainable adapter."""
    initial_adapter = config.get("initial_adapter_path")
    if initial_adapter:
        path = Path(initial_adapter)
        if not path.exists():
            raise FileNotFoundError(
                f"initial_adapter_path does not exist: {path}"
            )
        return peft_model_class.from_pretrained(
            model,
            str(path),
            is_trainable=True,
        )
    return fast_language_model.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0)),
        bias="none",
        random_state=int(config.get("seed", 3407)),
        use_rslora=False,
        loftq_config=None,
    )
