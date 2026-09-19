from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
    ImageGenerationNewsItem,
    ImageModelRequest,
    ImageModelResponse,
    OpenAIMovieNewsImageGenerator,
)


EXPECTED_NORMAL_PROMPT_VERSION = "movie_news_image_v5"
EXPECTED_FALLBACK_PROMPT_VERSION = (
    "movie_news_image_moderation_fallback_v7"
)


class _NeverCalledImageClient:
    """Client-заглушка: в этом тесте Image API не вызывается."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        raise AssertionError(
            "Image API не должен вызываться."
        )


def main() -> int:
    if (
        OPENAI_IMAGE_PROMPT_VERSION
        != EXPECTED_NORMAL_PROMPT_VERSION
    ):
        raise AssertionError(
            "Unexpected normal prompt version: "
            f"{OPENAI_IMAGE_PROMPT_VERSION}"
        )

    if (
        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        != EXPECTED_FALLBACK_PROMPT_VERSION
    ):
        raise AssertionError(
            "Unexpected fallback prompt version: "
            f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
        )

    generator = OpenAIMovieNewsImageGenerator(
        client=_NeverCalledImageClient(),
        model_name="gpt-image-2",
        size="1024x1024",
        quality="medium",
    )

    items = (
        ImageGenerationNewsItem(
            position=1,
            news_id=1,
            title=(
                "Practical Magic 2 challenges "
                "Spider-Man: Brand New Day"
            ),
            summary=(
                "The release is compared with "
                "Spider-Man: Brand New Day at the box office."
            ),
        ),
        ImageGenerationNewsItem(
            position=2,
            news_id=2,
            title=(
                "Actor joins new X-Men film"
            ),
            summary=(
                "The actor has been cast in the new X-Men film."
            ),
        ),
        ImageGenerationNewsItem(
            position=3,
            news_id=3,
            title=(
                "Studio merger receives regulatory approval"
            ),
            summary=(
                "Two film companies received approval for the transaction."
            ),
        ),
    )

    normal_request = generator.build_request(
        items=items
    )

    required_normal_fragments = (
        "квадратную редакционную иллюстрацию",
        "НЕ добавляй на изображение",
        "годы жизни, даты рождения или смерти",
        "«1945–2026»",
        "САМОДЕЛЬНОГО ПОСТЕРА",
        "коллаж из двух или нескольких оригинальных мини-постеров",
        "не должен быть копией официального постера",
        '"title":"Practical Magic 2 challenges Spider-Man: Brand New Day"',
        '"summary":"The release is compared with Spider-Man: Brand New Day at the box office."',
    )

    for fragment in required_normal_fragments:
        if fragment not in normal_request.prompt:
            raise AssertionError(
                "NORMAL v5 missing fragment: "
                f"{fragment!r}"
            )

    generator.set_moderation_safe_editorial_fallback(
        True
    )

    fallback_request = generator.build_request(
        items=items
    )

    required_fallback_fragments = (
        "квадратную редакционную иллюстрацию",
        "НЕ добавляй на изображение",
        "годы жизни, даты рождения или смерти",
        "«1945–2026»",
        '"mode":"safer_title_poster_editorial_v6"',
        '"title":"Practical Magic 2 challenges Spider-Man: Brand New Day"',
        '"summary":"The release is compared with Spider-Man: Brand New Day at the box office."',
        "Это НЕ режим абстрактных универсальных картинок",
        "сделай свой оригинальный постер, а не копию официального",
        "коллаж из нескольких",
    )

    for fragment in required_fallback_fragments:
        if fragment not in fallback_request.prompt:
            raise AssertionError(
                "Fallback v7 missing fragment: "
                f"{fragment!r}"
            )

    forbidden_fallback_fragments = (
        '"semantic_visual_brief":',
        "semantic_visual_brief_v5",
    )

    for fragment in forbidden_fallback_fragments:
        if fragment in fallback_request.prompt:
            raise AssertionError(
                "Fallback v7 still contains legacy semantic fallback: "
                f"{fragment!r}"
            )

    if fallback_request.prompt == normal_request.prompt:
        raise AssertionError(
            "Fallback prompt должен отличаться от NORMAL prompt."
        )

    print(
        "NORMAL v5 safe custom-poster strategy: OK"
    )
    print(
        "NORMAL v5 multi-film poster collage strategy: OK"
    )
    print(
        "Fallback v7 keeps factual title/summary: OK"
    )
    print(
        "Fallback v7 avoids legacy semantic abstraction: OK"
    )
    print(
        "OpenAI requests: not performed"
    )
    print(
        "Safe poster image prompt test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
