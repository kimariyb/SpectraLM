"""Tests for text-only response masking collator."""

from __future__ import annotations

import torch

from src.training.response_masking import validate_response_only_batch
from src.training.text_collator import TextResponseOnlyCollator


class FakeChatTokenizer:
    """Tiny reversible tokenizer with a Qwen-like chat template."""

    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.char_to_id = {"<pad>": 0}
        self.id_to_char = {0: ""}
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if kwargs.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n"
        return text

    def _encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char not in self.char_to_id:
                token_id = len(self.char_to_id)
                self.char_to_id[char] = token_id
                self.id_to_char[token_id] = char
            ids.append(self.char_to_id[char])
        return ids

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        encoded = [self._encode(text) for text in texts]
        max_len = max(len(row) for row in encoded)
        padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in encoded]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

    def decode(self, token_ids, *, skip_special_tokens: bool):
        text = "".join(self.id_to_char[int(token_id)] for token_id in token_ids)
        if skip_special_tokens:
            text = text.replace("<eos>", "")
        return text


def test_text_collator_masks_prompt_and_supervises_only_assistant_target() -> None:
    tokenizer = FakeChatTokenizer()
    collator = TextResponseOnlyCollator(tokenizer)
    features = [
        {
            "messages": [
                {"role": "system", "content": "System."},
                {"role": "user", "content": "Predict from NMR."},
                {"role": "assistant", "content": "CCO"},
            ]
        }
    ]

    batch = collator(features)

    stats = validate_response_only_batch(
        batch,
        tokenizer,
        expected_response="CCO",
    )
    assert stats["decoded_response"] == "CCO"
    assert stats["supervised_tokens"] == 3
    assert -100 in batch["labels"][0].tolist()
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)


def test_text_collator_rejects_non_prefix_chat_template() -> None:
    class BrokenTokenizer(FakeChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("add_generation_prompt"):
                return "different-prefix"
            return text

    collator = TextResponseOnlyCollator(BrokenTokenizer())
    features = [
        {
            "messages": [
                {"role": "user", "content": "Predict."},
                {"role": "assistant", "content": "CCO"},
            ]
        }
    ]

    try:
        collator(features)
    except RuntimeError as exc:
        assert "chat template prefix" in str(exc)
    else:
        raise AssertionError("Expected strict prefix validation to fail")
