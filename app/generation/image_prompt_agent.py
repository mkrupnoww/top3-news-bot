"""Планирование визуальной концепции без вызова Image API."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.generation.image_generator import (
    IMAGE_PROMPT_NORMAL_VERSION,
    IMAGE_PROMPT_RECOVERY_VERSION,
    ImageGenerationNewsItem,
)
from app.ranking.openai_usage import (
    calculate_openai_cost,
    extract_response_usage,
    get_model_pricing,
)


IMAGE_PROMPT_AGENT_VERSION = "openai_movie_news_image_prompt_agent_v1"
IMAGE_PROMPT_AGENT_PROMPT_VERSION = "movie_news_image_prompt_agent_v1"
MAX_IMAGE_PROMPT_BYTES = 4000
MAX_AGENT_OUTPUT_TOKENS = 5000
PromptMode = Literal[
    "normal_creative", "moderation_recovery", "safe_editorial_fallback"
]
PROMPT_DIRECTORY = Path(__file__).resolve().parents[2] / "prompts" / "prompts_image"


class ImageBlockPlan(BaseModel):
    """Краткое решение по одной новости, без chain-of-thought."""

    model_config = ConfigDict(extra="forbid", strict=True)
    position: int = Field(ge=1, le=3)
    news_id: int = Field(gt=0)
    strategy: str = Field(min_length=1, max_length=80)
    visual_concept: str = Field(min_length=1, max_length=500)
    avoid: list[str] = Field(max_length=8)


class ImagePromptPlan(BaseModel):
    """Строгий контракт готового короткого image prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)
    mode: PromptMode
    blocks: list[ImageBlockPlan] = Field(min_length=3, max_length=3)
    overall_style: str = Field(min_length=1, max_length=500)
    final_image_prompt: str = Field(min_length=1, max_length=2200)

    @model_validator(mode="after")
    def validate_plan(self) -> "ImagePromptPlan":
        if [block.position for block in self.blocks] != [1, 2, 3]:
            raise ValueError("План должен содержать блоки 1, 2, 3 в этом порядке.")
        if len({block.news_id for block in self.blocks}) != 3:
            raise ValueError("В плане должны быть три разные новости.")
        if not self.final_image_prompt.strip():
            raise ValueError("Image prompt не может состоять из пробелов.")
        if self.final_image_prompt != self.final_image_prompt.strip():
            raise ValueError("Exact image prompt не должен иметь внешних пробелов.")
        if len(self.final_image_prompt.encode("utf-8")) > MAX_IMAGE_PROMPT_BYTES:
            raise ValueError("Image prompt превышает 4000 UTF-8 bytes.")
        return self


@dataclass(frozen=True, slots=True)
class ImagePromptAgentResult:
    plan: ImagePromptPlan
    response_metadata: dict[str, Any]
    usage: dict[str, Any]
    cost: dict[str, Any]


class ImagePromptAgent:
    """Один Responses request на одну сохранённую попытку планирования."""

    def __init__(self, *, client: Any, model_name: str = "gpt-6.1-sol") -> None:
        self.client = client
        self.model_name = model_name
        get_model_pricing(model_name)

    def build_input(
        self,
        *,
        mode: PromptMode,
        items: tuple[ImageGenerationNewsItem, ...],
        post_text: str,
        generated_items: list[dict[str, Any]],
        attempt_number: int,
        previous: dict[str, Any] | None,
        editorial_comment: str | None,
        issues: tuple[str, ...],
    ) -> tuple[str, dict[str, Any]]:
        """Recovery-инструкции и примеры изолированы от NORMAL."""

        if mode not in {"normal_creative", "moderation_recovery", "safe_editorial_fallback"}:
            raise ValueError("Неизвестный режим Image Prompt Agent.")
        if mode != "normal_creative" and previous is None:
            raise ValueError("Recovery требует предыдущую попытку.")
        instructions = (PROMPT_DIRECTORY / "image-prompt-agent-normal.txt").read_text(
            encoding="utf-8"
        )
        payload: dict[str, Any] = {
            "mode": mode,
            "attempt_number": attempt_number,
            "news": [asdict(item) for item in items],
            "approved_post_text": post_text,
            "approved_generated_items": generated_items,
            "editorial_comment": editorial_comment,
            "issues": list(issues),
        }
        if mode != "normal_creative":
            instructions += "\n" + (
                PROMPT_DIRECTORY / "image-prompt-agent-recovery.txt"
            ).read_text(encoding="utf-8")
            payload["previous_attempt"] = previous
            payload["recovery_examples"] = (
                PROMPT_DIRECTORY / "image-prompt-moderation-recovery-examples.txt"
            ).read_text(encoding="utf-8")
        if mode == "safe_editorial_fallback":
            instructions += (
                "\nSAFE_EDITORIAL_FALLBACK: используй названия и подтверждённые цифры, "
                "типографику или нейтральные киноиндустриальные сцены. "
                "Сохрани конкретный смысл каждой новости."
            )
        return instructions, payload

    async def generate(
        self, *, instructions: str, payload: dict[str, Any]
    ) -> ImagePromptAgentResult:
        """Не использует web search, повторные запросы или Image API."""

        response = await self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            reasoning={"effort": "low"},
            max_output_tokens=MAX_AGENT_OUTPUT_TOKENS,
            text={"format": {
                "type": "json_schema",
                "name": "movie_news_image_prompt_plan",
                "strict": True,
                "schema": ImagePromptPlan.model_json_schema(),
            }},
            store=False,
        )
        usage = extract_response_usage(response)
        cost = calculate_openai_cost(usage, get_model_pricing(self.model_name))
        telemetry = {
            "usage": asdict(usage),
            "cost": {
                key: format(value, ".8f") if not isinstance(value, str) else value
                for key, value in asdict(cost).items()
            },
            "response_metadata": {
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "request_id": getattr(response, "_request_id", None),
                "status": getattr(response, "status", None),
            },
        }
        try:
            if getattr(response, "status", "completed") != "completed":
                raise ValueError("Responses API не завершил image prompt plan.")
            plan = ImagePromptPlan.model_validate_json(response.output_text)
            if plan.mode != payload["mode"]:
                raise ValueError("Режим ответа агента отличается от запроса.")
            if [block.news_id for block in plan.blocks] != [
                item["news_id"] for item in payload["news"]
            ]:
                raise ValueError("Агент изменил состав или порядок TOP-3.")
            previous = payload.get("previous_attempt")
            if previous and plan.final_image_prompt == previous.get("final_image_prompt"):
                raise ValueError("Recovery вернул точно тот же отклонённый prompt.")
            if previous and previous.get("structured_plan"):
                previous_strategies = [
                    block["strategy"] for block in previous["structured_plan"]["blocks"]
                ]
                if previous_strategies == [block.strategy for block in plan.blocks]:
                    raise ValueError("Recovery должен сменить визуальную strategy хотя бы одного блока.")
        except Exception as error:
            # Даже при некорректном structured output сохраняем реальную стоимость.
            error.image_prompt_telemetry = telemetry
            raise
        return ImagePromptAgentResult(plan=plan, **telemetry)
