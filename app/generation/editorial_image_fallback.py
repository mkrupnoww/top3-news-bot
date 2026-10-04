"""Локальная содержательная карточка TOP-3 после исчерпания Image API budget."""

from io import BytesIO
import json
import re
import time

from PIL import Image, ImageDraw, ImageFont

from app.generation.image_generator import (
    ImageGenerationNewsItem, ImageModelRequest, ImageModelResponse,
    OpenAIImageGenerationResult, OpenAIImageGeneratorMetadata,
)
from app.generation.openai_generator import POST_POSITION_MARKERS


EDITORIAL_FALLBACK_VERSION = "movie_news_local_editorial_fallback_v1"


def _wrapped_lines(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


class LocalEditorialImageGenerator:
    """Три конкретных заголовка вместо универсального абстрактного PNG."""

    def __init__(self, *, post_text: str | None = None):
        self.headlines = []
        if post_text is not None:
            for marker in POST_POSITION_MARKERS:
                match = re.search(re.escape(marker) + r"\s+\*\*(.+?)\*\*(?:\r?\n|$)", post_text)
                if match is None:
                    self.headlines = []
                    break
                self.headlines.append(match.group(1))

    def _title(self, item):
        return self.headlines[item.position-1] if len(self.headlines) == 3 else item.title

    @property
    def metadata(self):
        return OpenAIImageGeneratorMetadata(
            generator_name="local_editorial_image_generator",
            generator_version="local_editorial_image_generator_v1",
            prompt_version=EDITORIAL_FALLBACK_VERSION,
            model_name="local_editorial_title_card",
        )

    def build_request(self, *, items, editorial_comment=None, issues=()):
        return ImageModelRequest(
            model=self.metadata.model_name,
            prompt="Локальная редакционная карточка трёх выбранных киноновостей:\n" + json.dumps(
                [{"position": item.position, "news_id": item.news_id, "title": self._title(item)} for item in items],
                ensure_ascii=False,
            ),
            size="1024x1024", quality="medium", output_format="png",
            background="opaque", moderation="auto", n=1,
        )

    async def generate(self, *, items: tuple[ImageGenerationNewsItem, ...], editorial_comment=None, issues=()):
        request = self.build_request(items=items)
        image = Image.new("RGB", (1024, 1024), "#f4f5fa")
        draw = ImageDraw.Draw(image)
        heading = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
        draw.text((48, 34), "TOP-3 НОВОСТЕЙ КИНО", fill="#14223d", font=heading)
        colors = [("#ddeafd", "#215fc6"), ("#ffe4cc", "#ad4d09"), ("#e5ddfa", "#6c37ae")]
        for item, (background, accent) in zip(items, colors, strict=True):
            top = 128 + (item.position - 1) * 288
            draw.rounded_rectangle((32, top, 992, top + 268), radius=20, fill=background)
            draw.text((52, top + 22), str(item.position), fill=accent, font=heading)
            font_size = 38
            while True:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                lines = _wrapped_lines(draw, self._title(item), font, 820)
                if len(lines) * (font_size + 10) <= 225 or font_size <= 18:
                    break
                font_size -= 2
            max_lines = 225 // (font_size + 10)
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                while draw.textlength(lines[-1] + "…", font=font) > 820:
                    lines[-1] = lines[-1][:-1]
                lines[-1] += "…"
            for offset, line in enumerate(lines):
                draw.text((128, top + 22 + offset * (font_size + 10)), line, fill="#17233b", font=font)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return OpenAIImageGenerationResult(
            model_request=request,
            model_response=ImageModelResponse(
                image_bytes=buffer.getvalue(), created=int(time.time()), output_format="png",
                quality="medium", size="1024x1024", background="opaque", usage=None,
            ),
        )
