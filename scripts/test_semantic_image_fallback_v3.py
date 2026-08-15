import json

from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
    ImageGenerationNewsItem,
    ImageModelRequest,
    OpenAIMovieNewsImageGenerator,
    build_image_prompt,
)


class _NoNetworkImageClient:
    """Fake transport: этот тест не должен вызывать Image API."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ):
        raise AssertionError(
            "Image API не должен вызываться."
        )


ITEMS = (
    ImageGenerationNewsItem(
        position=1,
        news_id=1029,
        title=(
            "Marvel X-Men reboot gets "
            "May 2028 release date"
        ),
        summary=(
            "The X-Men reboot is a new "
            "superhero project planned "
            "for theatrical release."
        ),
    ),
    ImageGenerationNewsItem(
        position=2,
        news_id=1037,
        title="Frozen 3 first details revealed",
        summary=(
            "The animated sequel returns "
            "to a magical winter fantasy "
            "setting with snow and ice."
        ),
    ),
    ImageGenerationNewsItem(
        position=3,
        news_id=986,
        title=(
            "Mexico approves "
            "Paramount-Warner merger"
        ),
        summary=(
            "A regulator approved the merger "
            "between Paramount and Warner "
            "in a corporate cinema-industry deal."
        ),
    ),
)


def _extract_payload(
    prompt: str,
) -> dict[str, object]:
    """Извлекает последний JSON payload из prompt."""

    marker = (
        "============================================================\n"
        "ВХОДНЫЕ ДАННЫЕ\n"
        "============================================================\n\n"
    )

    if marker not in prompt:
        raise AssertionError(
            "В prompt отсутствует блок "
            "ВХОДНЫЕ ДАННЫЕ."
        )

    payload_text = prompt.rsplit(
        marker,
        1,
    )[1]

    payload = json.loads(
        payload_text
    )

    if not isinstance(payload, dict):
        raise AssertionError(
            "Prompt payload должен быть object."
        )

    return payload


def _brief(
    payload: dict[str, object],
    *,
    position: int,
) -> dict[str, object]:
    """Возвращает semantic_visual_brief заданной позиции."""

    top3 = payload.get("top3")

    if not isinstance(top3, list):
        raise AssertionError(
            "payload.top3 должен быть list."
        )

    for entry in top3:
        if (
            isinstance(entry, dict)
            and entry.get("position")
            == position
        ):
            brief = entry.get(
                "semantic_visual_brief"
            )

            if not isinstance(
                brief,
                dict,
            ):
                raise AssertionError(
                    "semantic_visual_brief "
                    "должен быть object."
                )

            return brief

    raise AssertionError(
        f"Позиция {position} не найдена."
    )


def main() -> int:
    """Проверяет semantic moderation fallback v3 без сети."""

    if (
        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        != "movie_news_image_moderation_fallback_v3"
    ):
        raise AssertionError(
            "Неожиданный fallback prompt version: "
            f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
        )

    normal_prompt = build_image_prompt(
        items=ITEMS,
        moderation_safe_editorial_fallback=False,
    )

    normal_lower = normal_prompt.casefold()

    for required_identity in (
        "x-men",
        "frozen 3",
        "paramount",
        "warner",
    ):
        if (
            required_identity
            not in normal_lower
        ):
            raise AssertionError(
                "Normal prompt неожиданно потерял "
                "исходный идентификатор: "
                f"{required_identity}"
            )

    print(
        "Normal prompt remains unchanged: OK"
    )

    fallback_prompt = build_image_prompt(
        items=ITEMS,
        moderation_safe_editorial_fallback=True,
    )

    fallback_lower = fallback_prompt.casefold()

    for forbidden_identity in (
        "x-men",
        "frozen 3",
        "paramount",
        "warner",
        "marvel",
    ):
        if forbidden_identity in fallback_lower:
            raise AssertionError(
                "Fallback prompt содержит "
                "исходный идентификатор: "
                f"{forbidden_identity}"
            )

    print(
        "Exact source identities removed: OK"
    )

    payload = _extract_payload(
        fallback_prompt
    )

    fallback_metadata = payload.get(
        "moderation_safe_editorial_fallback"
    )

    if not isinstance(
        fallback_metadata,
        dict,
    ):
        raise AssertionError(
            "Fallback metadata отсутствует."
        )

    if (
        fallback_metadata.get("enabled")
        is not True
    ):
        raise AssertionError(
            "Fallback metadata enabled != true."
        )

    if (
        fallback_metadata.get("mode")
        != "semantic_visual_brief_v3"
    ):
        raise AssertionError(
            "Неожиданный fallback mode."
        )

    print(
        "Fallback payload version: OK"
    )

    superhero = _brief(
        payload,
        position=1,
    )

    superhero_text = json.dumps(
        superhero,
        ensure_ascii=False,
    ).casefold()

    for expected in (
        "супергерой",
        "футурист",
        "энерг",
    ):
        if expected not in superhero_text:
            raise AssertionError(
                "Superhero semantic brief "
                "потерял ожидаемый смысл: "
                f"{expected}"
            )

    if (
        "костюм"
        not in superhero_text
        or "эмблем"
        not in superhero_text
    ):
        raise AssertionError(
            "Superhero brief не содержит "
            "identity-safe ограничений."
        )

    print(
        "Superhero semantic meaning preserved: OK"
    )

    winter = _brief(
        payload,
        position=2,
    )

    winter_text = json.dumps(
        winter,
        ensure_ascii=False,
    ).casefold()

    for expected in (
        "зим",
        "снег",
        "лёд",
        "северное сияние",
        "голуб",
    ):
        if expected not in winter_text:
            raise AssertionError(
                "Winter semantic brief "
                "потерял ожидаемый смысл: "
                f"{expected}"
            )

    print(
        "Winter fantasy semantic meaning preserved: OK"
    )

    business = _brief(
        payload,
        position=3,
    )

    business_text = json.dumps(
        business,
        ensure_ascii=False,
    ).casefold()

    for expected in (
        "корпоратив",
        "регулятор",
        "документ",
        "переговор",
    ):
        if expected not in business_text:
            raise AssertionError(
                "Business semantic brief "
                "потерял ожидаемый смысл: "
                f"{expected}"
            )

    print(
        "Business semantic meaning preserved: OK"
    )

    generator = (
        OpenAIMovieNewsImageGenerator(
            client=_NoNetworkImageClient(),
            model_name="gpt-image-2",
            size="1024x1536",
        )
    )

    if (
        generator.metadata.prompt_version
        != OPENAI_IMAGE_PROMPT_VERSION
    ):
        raise AssertionError(
            "Normal generator metadata "
            "prompt_version неверна."
        )

    generator.set_moderation_safe_editorial_fallback(
        True
    )

    if (
        generator.metadata.prompt_version
        != OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
    ):
        raise AssertionError(
            "Fallback generator metadata "
            "prompt_version неверна."
        )

    request = generator.build_request(
        items=ITEMS
    )

    if (
        request.prompt
        != fallback_prompt
    ):
        raise AssertionError(
            "Generator fallback request "
            "не совпадает с build_image_prompt()."
        )

    print(
        "Generator switches to fallback v3: OK"
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
        "Semantic image fallback v3 test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )