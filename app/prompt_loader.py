from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts"


def load_prompt(
    relative_path: str,
) -> str:
    """Читает обязательный текстовый промпт из каталога prompts."""

    normalized_path = relative_path.strip()

    if not normalized_path:
        raise ValueError(
            "relative_path не может быть пустым."
        )

    candidate_path = (
        PROMPTS_ROOT / normalized_path
    ).resolve()

    prompts_root = PROMPTS_ROOT.resolve()

    if (
        candidate_path != prompts_root
        and prompts_root not in candidate_path.parents
    ):
        raise ValueError(
            "Путь к промпту выходит за пределы "
            "каталога prompts."
        )

    if not candidate_path.is_file():
        raise FileNotFoundError(
            "Файл промпта не найден: "
            f"{candidate_path}"
        )

    prompt_text = candidate_path.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt_text:
        raise ValueError(
            "Файл промпта не может быть пустым: "
            f"{candidate_path}"
        )

    return prompt_text