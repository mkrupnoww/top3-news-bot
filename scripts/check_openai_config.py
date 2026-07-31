import argparse

import openai

from app.config import get_settings


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры проверки."""

    parser = argparse.ArgumentParser(
        description=(
            "Проверяет локальную конфигурацию "
            "OpenAI без обращения к API."
        ),
    )

    parser.add_argument(
        "--require-key",
        action="store_true",
        help=(
            "Завершить проверку с ошибкой, "
            "если OPENAI_API_KEY не задан."
        ),
    )

    return parser.parse_args()


def main(
    arguments: argparse.Namespace,
) -> int:
    """
    Проверяет настройки без сетевого запроса.
    """

    settings = get_settings()

    api_key_configured = (
        settings.openai_api_key is not None
        and bool(
            settings
            .openai_api_key
            .get_secret_value()
            .strip()
        )
    )

    print(
        "OpenAI configuration check completed"
    )
    print(
        f"openai_sdk_version="
        f"{openai.__version__}"
    )
    print(
        f"ranking_model="
        f"{settings.openai_ranking_model}"
    )
    print(
        f"timeout_seconds="
        f"{settings.openai_timeout_seconds}"
    )
    print(
        f"max_retries="
        f"{settings.openai_max_retries}"
    )
    print(
        "api_key_configured="
        f"{str(api_key_configured).lower()}"
    )
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")

    if (
        arguments.require_key
        and not api_key_configured
    ):
        print(
            "OpenAI configuration check: FAILED"
        )
        print(
            "OPENAI_API_KEY is not configured."
        )
        return 1

    print("OpenAI configuration check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(parse_arguments())
    )