import json

from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
    ImageGenerationNewsItem,
    ImageModelRequest,
    OpenAIMovieNewsImageGenerator,
    build_image_prompt,
)


EXPECTED_NORMAL_PROMPT_VERSION = (
    "movie_news_image_v3"
)
EXPECTED_FALLBACK_PROMPT_VERSION = (
    "movie_news_image_moderation_fallback_v5"
)
EXPECTED_FALLBACK_MODE = (
    "semantic_visual_brief_v5"
)


class _NoNetworkImageClient:
    """Fake transport: тест не должен вызывать Image API."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ):
        raise AssertionError(
            "Image API не должен вызываться."
        )


BASE_ITEMS = (
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


MONSTROPOLIS_ITEM = ImageGenerationNewsItem(
    position=1,
    news_id=1075,
    title=(
        "‘Monsters, Inc.’ Themed Land Will Open "
        "at Disney World in 2027"
    ),
    summary=(
        "The ‘Monsters, Inc.’ themed land, "
        "Monstropolis, is set to open at "
        "Disney’s Hollywood Studios at Disney "
        "World in 2027. The news was announced "
        "Saturday at D23. Vice President Creative, "
        "Walt Disney World Portfolio at Walt Disney "
        "Imagineering, Michael Hundgen told Variety "
        "the team is excited to bring Monstropolis "
        "to life for fans. The themed land is "
        "currently under construction."
    ),
)


TODAY_2026_08_21_ITEMS = (
    ImageGenerationNewsItem(
        position=1,
        news_id=1537,
        title=(
            "Крис Хансен отказался от условий A24 "
            "и не увидел Primetime"
        ),
        summary=(
            "Журналист Крис Хансен приехал в A24 "
            "на показ Primetime Лэнса Оппенхайма, "
            "где его играет Роберт Паттинсон. "
            "Но Хансен отказался подписать предложенное "
            "соглашение и ушёл, так и не увидев фильм."
        ),
    ),
    ImageGenerationNewsItem(
        position=2,
        news_id=1487,
        title=(
            "The Odyssey побила рекорд R-rated "
            "фильмов в прокате"
        ),
        summary=(
            "Приключенческий фильм собрал рекордную "
            "мировую кассу и стал крупнейшим "
            "R-rated релизом."
        ),
    ),
    ImageGenerationNewsItem(
        position=3,
        news_id=1528,
        title=(
            "Джафар Джексон снова сыграет "
            "Майкла Джексона в сиквеле"
        ),
        summary=(
            "Джафар Джексон вновь сыграет дядю, "
            "Майкла Джексона, в продолжении "
            "музыкального байопика Michael. "
            "Сиквел затронет взгляд певца на "
            "обвинения в сексуальном насилии "
            "над детьми."
        ),
    ),
)


def _neutral_item(
    *,
    position: int,
    news_id: int,
) -> ImageGenerationNewsItem:
    return ImageGenerationNewsItem(
        position=position,
        news_id=news_id,
        title="A new screen project is discussed",
        summary=(
            "The project remains in development "
            "with further information expected later."
        ),
    )


def _extract_payload(
    prompt: str,
) -> dict[str, object]:
    """Извлекает JSON payload из fallback prompt."""

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
    """Возвращает semantic_visual_brief позиции."""

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


def _brief_for_item(
    item: ImageGenerationNewsItem,
) -> dict[str, object]:
    """Строит fallback brief только для проверяемой позиции 1."""

    if item.position != 1:
        raise ValueError(
            "Regression item должен иметь position=1."
        )

    items = (
        item,
        _neutral_item(
            position=2,
            news_id=900002,
        ),
        _neutral_item(
            position=3,
            news_id=900003,
        ),
    )

    prompt = build_image_prompt(
        items=items,
        moderation_safe_editorial_fallback=True,
    )

    return _brief(
        _extract_payload(prompt),
        position=1,
    )


def _brief_text(
    brief: dict[str, object],
) -> str:
    return json.dumps(
        brief,
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()


def _assert_not_contains(
    text: str,
    forbidden: tuple[str, ...],
    *,
    context: str,
) -> None:
    for value in forbidden:
        if value.casefold() in text:
            raise AssertionError(
                f"{context}: найден ложный "
                f"semantic signal {value!r}"
            )


def main() -> int:
    """Проверяет semantic fallback v5 и marker matching без сети."""

    if (
        OPENAI_IMAGE_PROMPT_VERSION
        != EXPECTED_NORMAL_PROMPT_VERSION
    ):
        raise AssertionError(
            "Неожиданная normal prompt version: "
            f"{OPENAI_IMAGE_PROMPT_VERSION}"
        )

    if (
        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        != EXPECTED_FALLBACK_PROMPT_VERSION
    ):
        raise AssertionError(
            "Неожиданная fallback prompt version: "
            f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
        )

    # ------------------------------------------------------------------
    # Normal image mode v3 сохраняет исходные идентификаторы.
    # ------------------------------------------------------------------

    normal_prompt = build_image_prompt(
        items=BASE_ITEMS,
        moderation_safe_editorial_fallback=False,
    )

    normal_lower = normal_prompt.casefold()

    for required_identity in (
        "x-men",
        "frozen 3",
        "paramount",
        "warner",
    ):
        if required_identity not in normal_lower:
            raise AssertionError(
                "Normal prompt неожиданно потерял "
                "исходный идентификатор: "
                f"{required_identity}"
            )

    for required_instruction in (
        "person-centric",
        "центральный человек",
        "не визуализируй предполагаемое",
    ):
        if required_instruction not in normal_lower:
            raise AssertionError(
                "Normal prompt v3 потерял "
                "person-centric правило: "
                f"{required_instruction!r}"
            )

    print(
        "Normal prompt v3 identity/person priority: OK"
    )

    # ------------------------------------------------------------------
    # Fallback identity stripping и версия v5.
    # ------------------------------------------------------------------

    fallback_prompt = build_image_prompt(
        items=BASE_ITEMS,
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
        != EXPECTED_FALLBACK_MODE
    ):
        raise AssertionError(
            "Неожиданный fallback mode: "
            f"{fallback_metadata.get('mode')!r}"
        )

    print(
        "Fallback v5 identity and mode: OK"
    )

    # ------------------------------------------------------------------
    # Существующие положительные semantic-сигналы не теряем.
    # ------------------------------------------------------------------

    superhero_text = _brief_text(
        _brief(
            payload,
            position=1,
        )
    )

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

    winter_text = _brief_text(
        _brief(
            payload,
            position=2,
        )
    )

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

    business_text = _brief_text(
        _brief(
            payload,
            position=3,
        )
    )

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
        "Existing positive semantic signals preserved: OK"
    )

    # ------------------------------------------------------------------
    # Production regression 2026-08-21:
    # person-centric screening + sensitive musical biopic.
    # ------------------------------------------------------------------

    today_fallback_prompt = build_image_prompt(
        items=TODAY_2026_08_21_ITEMS,
        moderation_safe_editorial_fallback=True,
    )

    today_fallback_lower = (
        today_fallback_prompt.casefold()
    )

    for forbidden_identity in (
        "крис хансен",
        "a24",
        "primetime",
        "лэнс оппенхайм",
        "роберт паттинсон",
        "джафар джексон",
        "майкл джексон",
        "lionsgate",
    ):
        if forbidden_identity in today_fallback_lower:
            raise AssertionError(
                "Fallback v5 восстановил "
                "идентификатор production news: "
                f"{forbidden_identity!r}"
            )

    today_payload = _extract_payload(
        today_fallback_prompt
    )

    screening_text = _brief_text(
        _brief(
            today_payload,
            position=1,
        )
    )

    for expected in (
        "кинопоказ",
        "централь",
        "документ",
        "отказ",
        "журналист",
    ):
        if expected not in screening_text:
            raise AssertionError(
                "Screening/person semantic brief "
                "потерял production-смысл: "
                f"{expected}"
            )

    music_biopic_text = _brief_text(
        _brief(
            today_payload,
            position=3,
        )
    )

    for expected in (
        "музыкаль",
        "биограф",
        "централь",
        "микрофон",
        "софит",
        "чувствитель",
        "профессиональ",
    ):
        if expected not in music_biopic_text:
            raise AssertionError(
                "Music-biopic semantic brief "
                "потерял production-смысл: "
                f"{expected}"
            )

    _assert_not_contains(
        music_biopic_text,
        (
            "научная фантастика",
            "футуристическая технологическая",
            "космическая",
        ),
        context="2026-08-21 music biopic",
    )

    print(
        "2026-08-21 screening person semantics: OK"
    )
    print(
        "2026-08-21 sensitive music-biopic semantics: OK"
    )

    # ------------------------------------------------------------------
    # Production regression 2026-08-16:
    # "Vice President" не должен срабатывать как marker "ice".
    # ------------------------------------------------------------------

    monstropolis = _brief_for_item(
        MONSTROPOLIS_ITEM
    )
    monstropolis_text = _brief_text(
        monstropolis
    )

    _assert_not_contains(
        monstropolis_text,
        (
            "семейная сказочная зимняя тема",
            "зимний природный пейзаж",
            "снег и лёд",
            "северное сияние",
            "ледяные голубые",
            "магическая и зимняя",
        ),
        context="Monstropolis/Vice President",
    )

    for expected in (
        "тематическом парке",
        "развлекательной зоне",
        "декоративные фасады",
        "аттракционные конструкции",
    ):
        if expected not in monstropolis_text:
            raise AssertionError(
                "Theme-park semantic brief "
                "потерял ожидаемый смысл: "
                f"{expected}"
            )

    print(
        "Vice President does not match ice: OK"
    )
    print(
        "Generic theme-park semantics preserved: OK"
    )

    # ------------------------------------------------------------------
    # Настоящее отдельное слово "ice" по-прежнему срабатывает.
    # ------------------------------------------------------------------

    positive_ice = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910001,
            title=(
                "Winter fantasy project reveals "
                "new details"
            ),
            summary=(
                "The story takes place among "
                "snow and ice in a winter landscape."
            ),
        )
    )

    positive_ice_text = _brief_text(
        positive_ice
    )

    for expected in (
        "зим",
        "снег",
        "лёд",
    ):
        if expected not in positive_ice_text:
            raise AssertionError(
                "Настоящий winter/ice signal "
                "не был распознан: "
                f"{expected}"
            )

    print(
        "Standalone winter and ice markers still match: OK"
    )

    # ------------------------------------------------------------------
    # Дополнительные ложные substring-совпадения старой реализации.
    # ------------------------------------------------------------------

    artificial_intelligence = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910002,
            title=(
                "Фильм исследует искусственный "
                "интеллект"
            ),
            summary=(
                "Научная фантастика рассказывает "
                "о развитии искусственного интеллекта."
            ),
        )
    )
    artificial_text = _brief_text(
        artificial_intelligence
    )

    if (
        "корпоративное или регуляторное"
        in artificial_text
    ):
        raise AssertionError(
            "Русское 'искусственный' ложно "
            "сработало как business marker 'иск'."
        )

    if "научная фантастика" not in artificial_text:
        raise AssertionError(
            "Science-fiction signal потерян "
            "после исправления marker matching."
        )

    person = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910003,
            title=(
                "Человек киноиндустрии рассказал "
                "о новом проекте"
            ),
            summary=(
                "Автор рассказал о современной "
                "работе над экранным проектом."
            ),
        )
    )
    person_text = _brief_text(
        person
    )

    _assert_not_contains(
        person_text,
        (
            "историческая или эпохальная тема",
            "историческая архитектура",
        ),
        context="человек/век",
    )

    shop = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910004,
            title="Съёмочная группа посетила магазин",
            summary=(
                "Современный проект использует "
                "обычную городскую локацию."
            ),
        )
    )
    shop_text = _brief_text(
        shop
    )

    _assert_not_contains(
        shop_text,
        (
            "фэнтези и сказочное приключение",
            "сказочная",
        ),
        context="магазин/маг",
    )

    frost = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910005,
            title="История разворачивается в сильный мороз",
            summary=(
                "Зимняя сцена происходит на холодной "
                "городской улице."
            ),
        )
    )
    frost_text = _brief_text(
        frost
    )

    if "зим" not in frost_text:
        raise AssertionError(
            "Настоящий marker 'мороз' "
            "не распознан как winter."
        )

    _assert_not_contains(
        frost_text,
        (
            "морское побережье",
            "морской горизонт",
            "парусного судна",
        ),
        context="мороз/мор",
    )

    broadcast = _brief_for_item(
        ImageGenerationNewsItem(
            position=1,
            news_id=910006,
            title="Studio announces broadcast plans",
            summary=(
                "The company discussed a new "
                "broadcast window for the project."
            ),
        )
    )
    broadcast_text = _brief_text(
        broadcast
    )

    _assert_not_contains(
        broadcast_text,
        (
            "новость о кастинге",
            "будущего участия в кинопроекте",
        ),
        context="broadcast/cast",
    )

    print(
        "Known substring collision regressions: OK"
    )

    # ------------------------------------------------------------------
    # Runtime generator использует новую fallback identity.
    # ------------------------------------------------------------------

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
        != EXPECTED_FALLBACK_PROMPT_VERSION
    ):
        raise AssertionError(
            "Fallback generator metadata "
            "prompt_version неверна."
        )

    request = generator.build_request(
        items=BASE_ITEMS
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
        "Generator switches to fallback v5: OK"
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
        "Semantic marker matching v5 test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
