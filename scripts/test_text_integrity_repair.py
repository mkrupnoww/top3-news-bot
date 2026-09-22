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
    PostTextLengthOverflowError,
    _canonicalize_generated_payload_post_text,
    build_top3_post_text,
)
from app.generation.openai_pipeline import (
    _combine_generation_results,
    _run_integrity_repairs_if_needed,
)
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
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


def _overflow_payload() -> OpenAIGeneratedPostPayload:
    """
    Строит точный regression fixture:

    canonical post после trailer metadata имеет длину
    MAXIMUM_POST_LENGTH + 1 — тот же класс ошибки,
    который остановил daily workflow 2026-09-22.
    """

    base_body = "Т."

    def build_items(
        body_lengths: tuple[int, int, int],
    ) -> list[OpenAIGeneratedNewsPayload]:
        result = []

        for position, body_length in zip(
            (1, 2, 3),
            body_lengths,
            strict=True,
        ):
            if body_length < 2 or body_length > 210:
                raise AssertionError(
                    "Invalid regression body length: "
                    f"{body_length}"
                )

            result.append(
                OpenAIGeneratedNewsPayload(
                    position=position,
                    news_id=200 + position,
                    headline="З" * 70,
                    body=(
                        ("Т" * (body_length - 1))
                        + "."
                    ),
                    official_trailer_url=(
                        "https://www.youtube.com/"
                        f"watch?v=testvideo0{position}"
                    ),
                    official_trailer_channel_name=(
                        "Test Studio"
                    ),
                )
            )

        return result

    base_items = build_items(
        (
            len(base_body),
            len(base_body),
            len(base_body),
        )
    )

    base_post_text = build_top3_post_text(
        base_items
    )

    required_extra = (
        MAXIMUM_POST_LENGTH
        + 1
        - len(base_post_text)
    )

    if required_extra <= 0:
        raise AssertionError(
            "Base overflow fixture is already too long."
        )

    body_lengths = [
        len(base_body),
        len(base_body),
        len(base_body),
    ]

    for index in range(3):
        capacity = 210 - body_lengths[index]
        added = min(required_extra, capacity)

        body_lengths[index] += added
        required_extra -= added

        if required_extra == 0:
            break

    if required_extra != 0:
        raise AssertionError(
            "Cannot construct exact +1 overflow fixture."
        )

    payload_items = build_items(
        (
            body_lengths[0],
            body_lengths[1],
            body_lengths[2],
        )
    )

    return OpenAIGeneratedPostPayload(
        post_text="Допустимый модельный черновик.",
        items=payload_items,
    )


def _priority_overflow_payload() -> tuple[
    OpenAIGeneratedPostPayload,
    str,
]:
    """
    Строит +1 overflow для проверки приоритета compaction.

    У первой новости body содержит два завершённых
    предложения. Удаление последнего предложения сокращает
    текст сильнее, чем headline fallback. Старая логика,
    выбирающая минимальное сокращение без учёта типа
    кандидата, поэтому выбрала бы headline fallback.
    """

    first_sentence = ("П" * 48) + "."
    first_body = (
        first_sentence
        + " "
        + ("В" * 159)
        + "."
    )

    if len(first_body) != 210:
        raise AssertionError(
            "Priority regression body must have "
            "exactly 210 characters."
        )

    def build_item(
        *,
        position: int,
        body: str,
    ) -> OpenAIGeneratedNewsPayload:
        return OpenAIGeneratedNewsPayload(
            position=position,
            news_id=300 + position,
            headline="З" * 70,
            body=body,
            official_trailer_url=(
                "https://www.youtube.com/"
                f"watch?v=priority0{position}"
            ),
            official_trailer_channel_name=(
                "Priority Test Studio"
            ),
        )

    base_items = [
        build_item(
            position=1,
            body=first_body,
        ),
        build_item(
            position=2,
            body="Т.",
        ),
        build_item(
            position=3,
            body="Т.",
        ),
    ]

    base_post_text = build_top3_post_text(
        base_items
    )

    required_extra = (
        MAXIMUM_POST_LENGTH
        + 1
        - len(base_post_text)
    )

    if required_extra <= 0:
        raise AssertionError(
            "Priority regression base is already "
            "too long."
        )

    trailing_body_lengths = [2, 2]

    for index in range(2):
        capacity = (
            210
            - trailing_body_lengths[index]
        )
        added = min(
            required_extra,
            capacity,
        )

        trailing_body_lengths[index] += added
        required_extra -= added

        if required_extra == 0:
            break

    if required_extra != 0:
        raise AssertionError(
            "Cannot construct exact +1 priority "
            "overflow fixture."
        )

    payload_items = [
        base_items[0],
        build_item(
            position=2,
            body=(
                ("Т" * (
                    trailing_body_lengths[0] - 1
                ))
                + "."
            ),
        ),
        build_item(
            position=3,
            body=(
                ("Т" * (
                    trailing_body_lengths[1] - 1
                ))
                + "."
            ),
        ),
    ]

    return (
        OpenAIGeneratedPostPayload(
            post_text="Допустимый модельный черновик.",
            items=payload_items,
        ),
        first_sentence,
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

    overflow_payload = _overflow_payload()

    try:
        build_top3_post_text(
            overflow_payload.items
        )
    except PostTextLengthOverflowError as error:
        if error.actual_length != (
            MAXIMUM_POST_LENGTH + 1
        ):
            raise AssertionError(
                "Overflow fixture must reproduce "
                "exactly MAXIMUM_POST_LENGTH + 1: "
                f"actual={error.actual_length}"
            )
    else:
        raise AssertionError(
            "Overflow fixture did not overflow."
        )

    deferred_payload = (
        _canonicalize_generated_payload_post_text(
            overflow_payload
        )
    )

    if (
        deferred_payload.post_text
        != overflow_payload.post_text
    ):
        raise AssertionError(
            "Overflow canonicalization must defer "
            "to integrity recovery."
        )

    overflow_issues = (
        validate_generated_post_integrity(
            deferred_payload
        )
    )

    if not any(
        "Не удалось канонически собрать post_text"
        in issue
        for issue in overflow_issues
    ):
        raise AssertionError(
            "Integrity gate must detect canonical "
            "post overflow."
        )

    compacted_payload = (
        build_deterministic_integrity_fallback(
            deferred_payload
        )
    )

    if (
        len(compacted_payload.post_text)
        > MAXIMUM_POST_LENGTH
    ):
        raise AssertionError(
            "Deterministic compaction must fit "
            "MAXIMUM_POST_LENGTH."
        )

    if validate_generated_post_integrity(
        compacted_payload
    ):
        raise AssertionError(
            "Compacted overflow payload must pass "
            "integrity gate."
        )

    if not any(
        len(compacted.body) < len(original.body)
        for compacted, original in zip(
            compacted_payload.items,
            overflow_payload.items,
            strict=True,
        )
    ):
        raise AssertionError(
            "Overflow recovery must shorten at "
            "least one body."
        )

    print(
        "Canonical post overflow deferred and "
        "deterministically compacted: OK"
    )

    priority_payload, expected_first_sentence = (
        _priority_overflow_payload()
    )

    try:
        build_top3_post_text(
            priority_payload.items
        )
    except PostTextLengthOverflowError as error:
        if error.actual_length != (
            MAXIMUM_POST_LENGTH + 1
        ):
            raise AssertionError(
                "Priority overflow fixture must "
                "reproduce exactly "
                "MAXIMUM_POST_LENGTH + 1: "
                f"actual={error.actual_length}"
            )
    else:
        raise AssertionError(
            "Priority overflow fixture did not "
            "overflow."
        )

    priority_compacted = (
        build_deterministic_integrity_fallback(
            priority_payload
        )
    )

    if (
        priority_compacted.items[0].body
        != expected_first_sentence
    ):
        raise AssertionError(
            "Sentence-tail compaction must be used "
            "before headline fallback."
        )

    if (
        priority_compacted.items[1].body
        != priority_payload.items[1].body
        or priority_compacted.items[2].body
        != priority_payload.items[2].body
    ):
        raise AssertionError(
            "Priority compaction must not modify "
            "other bodies once sentence-tail "
            "reduction is sufficient."
        )

    if (
        priority_compacted.items[0].headline
        != priority_payload.items[0].headline
        or (
            priority_compacted.items[0]
            .official_trailer_url
            != priority_payload.items[0]
            .official_trailer_url
        )
        or (
            priority_compacted.items[0]
            .official_trailer_channel_name
            != priority_payload.items[0]
            .official_trailer_channel_name
        )
    ):
        raise AssertionError(
            "Priority compaction must preserve "
            "headline and trailer metadata."
        )

    if validate_generated_post_integrity(
        priority_compacted
    ):
        raise AssertionError(
            "Priority-compacted payload must pass "
            "integrity gate."
        )

    print(
        "Sentence-tail compaction has priority "
        "over headline fallback: OK"
    )

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
