"""Validation helpers for assistant-only text supervision."""

from __future__ import annotations

from typing import Any, Mapping


def assistant_response_text(sample: Mapping[str, Any]) -> str:
    """Extract the final assistant text target from one chat sample.

    Parameters
    ----------
    sample
        Dataset row containing a ``messages`` conversation.

    Returns
    -------
    str
        Non-empty text from the final assistant message.
    """
    for message in reversed(sample.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        for item in content:
            if item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    return text
    raise RuntimeError("Training sample has no non-empty assistant target.")


def _first_row(values: Any, name: str) -> list[int]:
    """Convert the first tensor or nested-sequence row to integer IDs."""
    if values is None or len(values) == 0:
        raise RuntimeError(f"Response-only preflight batch has no {name}.")
    row = values[0]
    if hasattr(row, "tolist"):
        row = row.tolist()
    return [int(value) for value in row]


def _normalize_decoded_text(text: str) -> str:
    """Ignore tokenizer-inserted whitespace when comparing exact targets."""
    return "".join(str(text).split())


def apply_non_thinking_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render a generation prompt with Qwen thinking explicitly disabled."""
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def validate_response_only_batch(
    batch: Mapping[str, Any],
    tokenizer: Any,
    *,
    expected_response: str,
) -> dict[str, int | str]:
    """Verify that collator labels contain only the assistant target.

    Parameters
    ----------
    batch
        One collated batch containing ``input_ids`` and masked ``labels``.
    tokenizer
        Processor or tokenizer exposing ``decode``.
    expected_response
        Exact assistant target from the uncollated sample.

    Returns
    -------
    dict[str, int | str]
        Token counts and decoded supervised response for logging.
    """
    input_ids = _first_row(batch.get("input_ids"), "input_ids")
    labels = _first_row(batch.get("labels"), "labels")
    if len(input_ids) != len(labels):
        raise RuntimeError(
            "Response-only preflight input_ids and labels have different lengths."
        )

    supervised_ids = [token_id for token_id in labels if token_id != -100]
    if not supervised_ids:
        raise RuntimeError(
            "Response-only masking produced no supervised assistant tokens; "
            "check the Qwen chat-template boundaries and max_seq_length."
        )

    decoder = getattr(tokenizer, "tokenizer", tokenizer)
    raw_decoded_response = decoder.decode(
        supervised_ids,
        skip_special_tokens=True,
    ).strip()
    if "<think>" in raw_decoded_response or "</think>" in raw_decoded_response:
        raise RuntimeError(
            "Response-only masking supervised thinking tokens; "
            "check enable_thinking=False and chat-template boundaries."
        )
    decoded_response = raw_decoded_response
    if _normalize_decoded_text(decoded_response) != _normalize_decoded_text(
        expected_response
    ):
        raise RuntimeError(
            "Response-only supervised text does not match the expected target. "
            f"Expected {expected_response!r}, decoded {raw_decoded_response!r}."
        )

    return {
        "sequence_tokens": len(input_ids),
        "supervised_tokens": len(supervised_ids),
        "decoded_response": decoded_response,
    }
