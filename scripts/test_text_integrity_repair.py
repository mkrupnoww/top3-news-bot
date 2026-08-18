from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    GenerationNewsItem,
    OpenAIGeneratedNewsPayload,
    OpenAIGeneratedPostPayload,
    OpenAIPostGenerationResult,
    build_top3_post_text,
)
from app.generation.openai_pipeline import (
    _combine_generation_results,
    _run_integrity_repairs_if_needed,
)
from app.generation.post_integrity import (
    body_has_suspicious_unterminated_tail,
    build_deterministic_integrity_fallback,
    validate_generated_post_integrity,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


class FakeRevisionGenerator:
    """Fake generator с очередью revision results."""

    def __init__(
        self,
        revision_results: tuple[
            OpenAIPostGenerationResult,
            ...,
        ],
    ) -> None:
        self._revision_results = revision_results
        self.revision_call_count = 0

    def build_revision_request(
        self,
        items: tuple[GenerationNewsItem, ...],
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> GenerationModelRequest:
        if not source_post_text.strip():
            raise AssertionError("source_post_text required")
        if not editorial_comment.strip():
            raise AssertionError("editorial_comment required")
        if not issues:
            raise AssertionError("issues required")

        return GenerationModelRequest(
            model="gpt-5.6-terra",
            instructions="revision",
            input_text="revision",
        )

    async def generate_prepared_revision_request(
        self,
        items: tuple[GenerationNewsItem, ...],
        request: GenerationModelRequest,
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> OpenAIPostGenerationResult:
        index = self.revision_call_count
        if index >= len(self._revision_results):
            raise AssertionError("unexpected extra revision")
        self.revision_call_count += 1
        return self._revision_results[index]


def _selection_items() -> tuple[
    GenerationNewsItem,
    GenerationNewsItem,
    GenerationNewsItem,
]:
    published_at = datetime(
        2026, 8, 18, 7, 30,
        tzinfo=timezone.utc,
    )

    result = []
    for position in (1, 2, 3):
        result.append(
            GenerationNewsItem(
                position=position,
                news_id=100 + position,
                title=f"Новость {position}",
                summary=f"Summary {position}",
                source_name="Test",
                source_url=f"https://example.com/{position}",
                source_published_at=published_at,
                individual_score=Decimal("5.000000"),
                selection_reason="Test",
            )
        )

    return result[0], result[1], result[2]


def _long_unterminated(prefix: str) -> str:
    target = 205
    suffix = " без финальной точки"
    filler_len = target - len(prefix) - len(suffix)
    if filler_len < 1:
        raise AssertionError("test prefix too long")
    return prefix + ("x" * filler_len) + suffix


def _result(
    bodies: tuple[str, str, str],
    *,
    token_base: int,
) -> OpenAIPostGenerationResult:
    payload_items = [
        OpenAIGeneratedNewsPayload(
            position=position,
            news_id=100 + position,
            headline=f"Заголовок {position}",
            body=bodies[position - 1],
        )
        for position in (1, 2, 3)
    ]

    post_text = build_top3_post_text(payload_items)
    payload = OpenAIGeneratedPostPayload(
        post_text=post_text,
        items=payload_items,
    )

    usage = OpenAITokenUsage(
        input_tokens=token_base,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=10,
        reasoning_tokens=0,
        total_tokens=token_base + 10,
    )

    cost = OpenAICostEstimate(
        model_name="gpt-5.6-terra",
        pricing_version="2026-07-31",
        regular_input_cost_usd=Decimal("0.00100000"),
        cached_input_cost_usd=Decimal("0"),
        cache_write_cost_usd=Decimal("0"),
        output_cost_usd=Decimal("0"),
        total_cost_usd=Decimal("0.00100000"),
    )

    return OpenAIPostGenerationResult(
        payload=payload,
        model_response=GenerationModelResponse(
            output_text=post_text,
            usage=usage,
            cost_estimate=cost,
        ),
    )


async def main() -> int:
    if body_has_suspicious_unterminated_tail(
        "Короткая нормальная фраза без точки"
    ):
        raise AssertionError(
            "Короткий body без точки не должен считаться truncation."
        )

    suspicious = _long_unterminated("Длинный текст ")
    if not body_has_suspicious_unterminated_tail(suspicious):
        raise AssertionError(
            "Длинный body у лимита без точки должен быть suspicious."
        )

    print("Targeted truncation heuristic: OK")

    primary = _result(
        (
            "Первая новость завершена.",
            "Вторая primary-новость завершена.",
            "Третья новость завершена.",
        ),
        token_base=100,
    )

    broken_self_review = _result(
        (
            "Первая новость завершена.",
            _long_unterminated("Self review "),
            "Третья новость завершена.",
        ),
        token_base=110,
    )

    broken_revision_1 = _result(
        (
            "Первая новость завершена.",
            _long_unterminated("Revision one "),
            "Третья новость завершена.",
        ),
        token_base=120,
    )

    broken_revision_2 = _result(
        (
            "Первая новость завершена.",
            _long_unterminated("Revision two "),
            "Третья новость завершена.",
        ),
        token_base=130,
    )

    fake = FakeRevisionGenerator(
        (broken_revision_1, broken_revision_2)
    )

    outcome = await _run_integrity_repairs_if_needed(
        fake,
        items=_selection_items(),
        initial_generation=broken_self_review,
        primary_generation=primary,
        max_revision_attempts=2,
    )

    if fake.revision_call_count != 2:
        raise AssertionError("Expected exactly two revision calls")

    if not outcome.used_deterministic_fallback:
        raise AssertionError("Expected deterministic fallback")

    if outcome.final_payload.items[1].body != (
        "Вторая primary-новость завершена."
    ):
        raise AssertionError(
            "When latest has no complete sentence, valid primary body must be reused."
        )

    if validate_generated_post_integrity(outcome.final_payload):
        raise AssertionError("Final fallback payload must pass integrity gate")

    print("Exhausted revisions use bounded local fail-safe: OK")

    complete_prefix = "Первое законченное предложение."
    unfinished = (
        complete_prefix
        + " "
        + (
            "х"
            * (
                205
                - len(complete_prefix)
                - 1
            )
        )
    )

    if len(unfinished) != 205:
        raise AssertionError(
            "Regression fixture должен иметь "
            "ровно 205 символов."
        )

    latest = _result(
        (
            "Первая новость завершена.",
            unfinished,
            "Третья новость завершена.",
        ),
        token_base=140,
    )

    deterministic = build_deterministic_integrity_fallback(
        latest.payload,
        fallback_payload=primary.payload,
    )

    if deterministic.items[1].body != complete_prefix:
        raise AssertionError(
            "Fallback must trim only the incomplete tail after last full sentence."
        )

    print("Deterministic fallback trims only incomplete tail: OK")

    combined = _combine_generation_results(
        (
            primary,
            broken_self_review,
            broken_revision_1,
            broken_revision_2,
        ),
        final_payload=outcome.final_payload,
    )

    usage = combined.model_response.usage
    if usage is None or usage.input_tokens != 460:
        raise AssertionError(
            "Usage must include only four actual model calls."
        )

    if combined.payload != outcome.final_payload:
        raise AssertionError("Combined result must keep fail-safe final payload")

    print("Model telemetry excludes local fallback double-counting: OK")

    print()
    print("Database changes=not_performed")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print("Text integrity fail-safe v2 test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
