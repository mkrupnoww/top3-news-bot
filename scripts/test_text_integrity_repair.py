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
    body_has_terminal_punctuation,
    validate_generated_post_integrity,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


class FakeRevisionGenerator:
    """Локальный fake для integrity repair."""

    def __init__(
        self,
        *,
        revision_result: (
            OpenAIPostGenerationResult
        ),
    ) -> None:
        self._revision_result = (
            revision_result
        )
        self.revision_call_count = 0
        self.last_source_post_text = ""
        self.last_editorial_comment = ""
        self.last_issues: tuple[str, ...] = ()

    def build_revision_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> GenerationModelRequest:
        self.last_source_post_text = (
            source_post_text
        )
        self.last_editorial_comment = (
            editorial_comment
        )
        self.last_issues = issues

        return GenerationModelRequest(
            model="gpt-5.6-terra",
            instructions=(
                "revision-instructions"
            ),
            input_text="revision-input",
        )

    async def generate_prepared_revision_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        request: GenerationModelRequest,
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> OpenAIPostGenerationResult:
        self.revision_call_count += 1

        if (
            source_post_text
            != self.last_source_post_text
        ):
            raise AssertionError(
                "source_post_text должен быть "
                "передан в revision без "
                "изменений."
            )

        if (
            editorial_comment
            != self.last_editorial_comment
        ):
            raise AssertionError(
                "editorial_comment в revision "
                "не совпадает с build."
            )

        if issues != self.last_issues:
            raise AssertionError(
                "issues в revision не "
                "совпадают с build."
            )

        if request.model != "gpt-5.6-terra":
            raise AssertionError(
                "Ожидалась тестовая модель "
                "gpt-5.6-terra."
            )

        return self._revision_result


def _make_selection_items() -> tuple[
    GenerationNewsItem,
    GenerationNewsItem,
    GenerationNewsItem,
]:
    published_at = datetime(
        2026,
        8,
        16,
        7,
        30,
        tzinfo=timezone.utc,
    )

    return (
        GenerationNewsItem(
            position=1,
            news_id=101,
            title="Новость 1",
            summary="Краткое описание 1",
            source_name="Source 1",
            source_url=(
                "https://example.com/1"
            ),
            source_published_at=published_at,
            individual_score=Decimal(
                "1.11"
            ),
            selection_reason=(
                "Причина выбора 1"
            ),
        ),
        GenerationNewsItem(
            position=2,
            news_id=102,
            title="Новость 2",
            summary="Краткое описание 2",
            source_name="Source 2",
            source_url=(
                "https://example.com/2"
            ),
            source_published_at=published_at,
            individual_score=Decimal(
                "2.22"
            ),
            selection_reason=(
                "Причина выбора 2"
            ),
        ),
        GenerationNewsItem(
            position=3,
            news_id=103,
            title="Новость 3",
            summary="Краткое описание 3",
            source_name="Source 3",
            source_url=(
                "https://example.com/3"
            ),
            source_published_at=published_at,
            individual_score=Decimal(
                "3.33"
            ),
            selection_reason=(
                "Причина выбора 3"
            ),
        ),
    )


def _make_generation_result(
    *,
    bodies: tuple[str, str, str],
    input_tokens: int,
    output_tokens: int,
    total_cost_usd: str,
    web_search_used: bool = False,
    web_search_call_count: int = 0,
    web_source_urls: tuple[str, ...] = (),
) -> OpenAIPostGenerationResult:
    payload_items = [
        OpenAIGeneratedNewsPayload(
            position=1,
            news_id=101,
            headline="Заголовок 1",
            body=bodies[0],
        ),
        OpenAIGeneratedNewsPayload(
            position=2,
            news_id=102,
            headline="Заголовок 2",
            body=bodies[1],
        ),
        OpenAIGeneratedNewsPayload(
            position=3,
            news_id=103,
            headline="Заголовок 3",
            body=bodies[2],
        ),
    ]

    post_text = build_top3_post_text(
        payload_items
    )

    payload = OpenAIGeneratedPostPayload(
        post_text=post_text,
        items=payload_items,
    )

    usage = OpenAITokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        total_tokens=(
            input_tokens + output_tokens
        ),
    )

    cost = OpenAICostEstimate(
        model_name="gpt-5.6-terra",
        pricing_version="2026-07-31",
        regular_input_cost_usd=Decimal(
            total_cost_usd
        ),
        cached_input_cost_usd=Decimal("0"),
        cache_write_cost_usd=Decimal("0"),
        output_cost_usd=Decimal("0"),
        total_cost_usd=Decimal(
            total_cost_usd
        ),
    )

    response = GenerationModelResponse(
        output_text=payload.post_text,
        usage=usage,
        cost_estimate=cost,
        web_search_used=web_search_used,
        web_search_call_count=(
            web_search_call_count
        ),
        web_source_urls=web_source_urls,
    )

    return OpenAIPostGenerationResult(
        payload=payload,
        model_response=response,
    )


async def _run() -> None:
    if body_has_terminal_punctuation(
        "Текст заканчивается точкой."
    ) is not True:
        raise AssertionError(
            "Точка должна считаться "
            "завершением."
        )

    if body_has_terminal_punctuation(
        'Текст заканчивается точкой.»'
    ) is not True:
        raise AssertionError(
            "Закрывающая кавычка после точки "
            "должна допускаться."
        )

    if body_has_terminal_punctuation(
        "Трейлер: "
        "[YouTube](https://example.com/trailer)"
    ) is not True:
        raise AssertionError(
            "Markdown-ссылка в конце должна "
            "допускаться."
        )

    if body_has_terminal_punctuation(
        "Оборванный текст строительcт"
    ) is not False:
        raise AssertionError(
            "Оборванный текст без знака "
            "завершения должен блокироваться."
        )

    print(
        "Terminal punctuation rules: OK"
    )

    items = _make_selection_items()

    primary_generation = (
        _make_generation_result(
            bodies=(
                "Первая новость завершена.",
                "Вторая новость завершена.",
                "Третья новость завершена.",
            ),
            input_tokens=100,
            output_tokens=50,
            total_cost_usd="0.01000000",
        )
    )

    broken_self_review = (
        _make_generation_result(
            bodies=(
                "Первая новость завершена.",
                "Вторая новость завершена.",
                "Тематическая зона находится "
                "в стадии строительcт",
            ),
            input_tokens=120,
            output_tokens=60,
            total_cost_usd="0.02000000",
            web_search_used=True,
            web_search_call_count=1,
            web_source_urls=(
                "https://example.com/source-a",
            ),
        )
    )

    fixed_revision = (
        _make_generation_result(
            bodies=(
                "Первая новость завершена.",
                "Вторая новость завершена.",
                "Тематическая зона находится "
                "в стадии строительства.",
            ),
            input_tokens=140,
            output_tokens=70,
            total_cost_usd="0.03000000",
            web_search_used=False,
            web_search_call_count=0,
            web_source_urls=(
                "https://example.com/source-b",
            ),
        )
    )

    issues = (
        validate_generated_post_integrity(
            broken_self_review.payload
        )
    )

    if not issues:
        raise AssertionError(
            "Обрезанный body должен был "
            "провалить integrity gate."
        )

    if not any(
        "position=3" in issue
        for issue in issues
    ):
        raise AssertionError(
            "Integrity gate должен указать "
            "на проблему в новости №3."
        )

    print(
        "Integrity gate catches truncated "
        "body: OK"
    )

    fake_generator = FakeRevisionGenerator(
        revision_result=fixed_revision
    )

    repaired_passes = (
        await _run_integrity_repairs_if_needed(
            fake_generator,
            items=items,
            initial_generation=(
                broken_self_review
            ),
            max_revision_attempts=2,
        )
    )

    if fake_generator.revision_call_count != 1:
        raise AssertionError(
            "Ожидалась ровно одна "
            "revision-попытка."
        )

    if len(repaired_passes) != 2:
        raise AssertionError(
            "Ожидались self-review и один "
            "revision-результат."
        )

    final_issues = (
        validate_generated_post_integrity(
            repaired_passes[-1].payload
        )
    )

    if final_issues:
        raise AssertionError(
            "Исправленный revision не должен "
            "проваливать integrity gate: "
            + "; ".join(final_issues)
        )

    print(
        "Automatic integrity revision "
        "repairs truncated body: OK"
    )

    combined = _combine_generation_results(
        (
            primary_generation,
            *repaired_passes,
        )
    )

    combined_usage = (
        combined.model_response.usage
    )

    combined_cost = (
        combined
        .model_response
        .cost_estimate
    )

    if combined_usage is None:
        raise AssertionError(
            "Комбинированный результат не "
            "содержит usage."
        )

    if combined_cost is None:
        raise AssertionError(
            "Комбинированный результат не "
            "содержит cost_estimate."
        )

    if combined_usage.input_tokens != 360:
        raise AssertionError(
            "input_tokens должны "
            "суммироваться по всем "
            "проходам."
        )

    if combined_usage.output_tokens != 180:
        raise AssertionError(
            "output_tokens должны "
            "суммироваться по всем "
            "проходам."
        )

    if combined_usage.total_tokens != 540:
        raise AssertionError(
            "total_tokens должны "
            "суммироваться по всем "
            "проходам."
        )

    if (
        combined_cost.total_cost_usd
        != Decimal("0.06000000")
    ):
        raise AssertionError(
            "Стоимость должна суммироваться "
            "по всем проходам."
        )

    if not (
        combined.model_response.web_search_used
    ):
        raise AssertionError(
            "Флаг web_search_used должен "
            "сохраняться после агрегации."
        )

    if (
        combined
        .model_response
        .web_search_call_count
        != 1
    ):
        raise AssertionError(
            "web_search_call_count должен "
            "суммироваться."
        )

    if (
        combined
        .model_response
        .web_source_urls
        != (
            "https://example.com/source-a",
            "https://example.com/source-b",
        )
    ):
        raise AssertionError(
            "web_source_urls должны "
            "дедуплицированно агрегироваться."
        )

    if combined.payload.items[2].body != (
        "Тематическая зона находится "
        "в стадии строительства."
    ):
        raise AssertionError(
            "Финальный payload должен брать "
            "body из последнего успешного "
            "revision."
        )

    print(
        "Combined usage and cost across "
        "primary + self-review + revision: OK"
    )

    print()
    print(
        "Database changes=not_performed"
    )
    print(
        "OpenAI requests=not_performed"
    )
    print(
        "Telegram requests=not_performed"
    )
    print(
        "Text integrity repair test: OK"
    )


if __name__ == "__main__":
    asyncio.run(_run())
