"""Text-only chat collator with assistant-response labels."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.training.response_masking import apply_non_thinking_chat_template


class TextResponseOnlyCollator:
    """Tokenize chat messages and mask all non-assistant target tokens."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        features: Sequence[Mapping[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """Collate message rows into ``input_ids``, ``attention_mask``, labels."""
        rendered: list[str] = []
        prefix_lengths: list[int] = []
        target_lengths: list[int] = []

        for feature in features:
            messages = list(feature["messages"])
            assistant = messages[-1]
            if assistant.get("role") != "assistant":
                raise RuntimeError("Training messages must end with assistant.")
            target_text = self._assistant_text(assistant)
            prompt_messages = messages[:-1]
            prefix = apply_non_thinking_chat_template(
                self.tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )
            full = apply_non_thinking_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=False,
            )
            if not full.startswith(prefix):
                raise RuntimeError(
                    "Text chat template prefix mismatch; cannot build "
                    "response-only labels from this tokenizer."
                )
            prefix_ids = self._encode_text(prefix)
            target_ids = self._encode_text(target_text)
            rendered.append(full)
            prefix_lengths.append(len(prefix_ids))
            target_lengths.append(len(target_ids))

        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self._pad_token_id()).long()

        labels = input_ids.clone()
        for row_idx, prefix_len in enumerate(prefix_lengths):
            target_end = min(prefix_len + target_lengths[row_idx], labels.shape[1])
            labels[row_idx, :prefix_len] = -100
            labels[row_idx, target_end:] = -100
            labels[row_idx, attention_mask[row_idx] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        return int(pad_token_id or 0)

    @staticmethod
    def _assistant_text(message: Mapping[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "".join(parts)
