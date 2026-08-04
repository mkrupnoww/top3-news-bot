from dataclasses import dataclass
import re
from urllib.parse import urlsplit

from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)


_WHITESPACE_PATTERN = re.compile(r"\s+")

_STRONG_CATEGORY_TERMS = (
    "movie",
    "movies",
    "movie news",
    "film",
    "films",
    "film news",
    "cinema",
    "box office",
    "filmmaker",
    "filmmaking",
    "screenplay",
    "screenwriting",
    "theatrical",
    "film festival",
    "imax",
)

_STRONG_URL_PARTS = (
    "/movies/",
    "/movie-news/",
    "/film/",
    "/films/",
    "/cinema/",
    "/box-office/",
)

_STRONG_TEXT_PATTERN = re.compile(
    r"\b("
    r"movie|movies|film|films|cinema|cinematic|imax|"
    r"box[ -]office|feature film|documentary|"
    r"filmmaker|filmmaking|screenplay|screenwriter|"
    r"theatrical release|film festival|"
    r"oscar|oscars|academy awards|"
    r"cannes|sundance|berlinale|locarno|"
    r"venice film festival|sarajevo film festival"
    r")\b",
    flags=re.IGNORECASE,
)

_REVIEW_CATEGORY_NAMES = frozenset(
    {
        "movie review",
        "movie reviews",
        "film review",
        "film reviews",
        "reviews",
    }
)

_REVIEW_TITLE_PATTERN = re.compile(
    r"\breview\b\s*[:—–-]",
    flags=re.IGNORECASE,
)

_EDITORIAL_RANKING_PATTERNS = (
    re.compile(
        r"\branked\s+from\s+"
        r"(?:worst|best)\s+to\s+"
        r"(?:best|worst)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bevery\b.{0,160}\branked\b",
        flags=re.IGNORECASE,
    ),
)

_FILM_INDUSTRY_ENTITIES = (
    "paramount",
    "warner bros",
    "warner brothers",
    "universal pictures",
    "sony pictures",
    "columbia pictures",
    "20th century studios",
    "twentieth century studios",
    "lionsgate",
    "a24",
    "focus features",
    "searchlight pictures",
    "dreamworks",
    "illumination",
    "pixar",
    "marvel studios",
    "dc studios",
    "amazon mgm studios",
    "mgm",
    "neon",
)

_FILM_INDUSTRY_ACTIONS = (
    "merger",
    "acquisition",
    "antitrust",
    "lawsuit",
    "deal",
    "distribution",
    "production",
    "release slate",
    "theatrical",
    "box office",
    "studio",
)


@dataclass(frozen=True, slots=True)
class CinemaRelevanceDecision:
    """Решение фильтра для одной публикации."""

    news_id: int
    source_code: str
    is_relevant: bool
    signals: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class CinemaRelevanceFilterResult:
    """Итог тематической фильтрации выборки."""

    selection: CandidateSelectionResult
    decisions: tuple[CinemaRelevanceDecision, ...]

    @property
    def input_count(self) -> int:
        """Количество публикаций до фильтрации."""

        return len(self.decisions)

    @property
    def included_count(self) -> int:
        """Количество оставленных публикаций."""

        return len(self.selection.candidates)

    @property
    def excluded_count(self) -> int:
        """Количество исключённых публикаций."""

        return self.input_count - self.included_count

    @property
    def excluded_news_ids(self) -> tuple[int, ...]:
        """Возвращает ID исключённых публикаций."""

        return tuple(
            decision.news_id
            for decision in self.decisions
            if not decision.is_relevant
        )


def _normalize_text(value: str | None) -> str:
    """Нормализует текст для поиска сигналов."""

    if value is None:
        return ""

    return _WHITESPACE_PATTERN.sub(
        " ",
        value,
    ).strip().casefold()


def _editorial_exclusion_signals(
    candidate: NewsCandidate,
) -> tuple[str, ...]:
    """
    Находит явные редакционные форматы без
    самостоятельного новостного инфоповода.

    Проверка применяется ко всем источникам,
    включая специализированные киноразделы.
    """

    signals: list[str] = []

    normalized_categories = tuple(
        _normalize_text(category)
        for category in candidate.categories
        if _normalize_text(category)
    )

    if any(
        category in _REVIEW_CATEGORY_NAMES
        for category in normalized_categories
    ):
        signals.append(
            "excluded_format:review_category"
        )

    if _REVIEW_TITLE_PATTERN.search(
        candidate.title
    ):
        signals.append(
            "excluded_format:review_title"
        )

    if any(
        pattern.search(candidate.title)
        for pattern in _EDITORIAL_RANKING_PATTERNS
    ):
        signals.append(
            "excluded_format:editorial_ranking"
        )

    return tuple(
        dict.fromkeys(signals)
    )


def _category_signals(
    candidate: NewsCandidate,
) -> tuple[str, ...]:
    """Находит сильные кино-сигналы в категориях."""

    signals: list[str] = []

    for category in candidate.categories:
        normalized_category = _normalize_text(category)

        if not normalized_category:
            continue

        if any(
            term in normalized_category
            for term in _STRONG_CATEGORY_TERMS
        ):
            signals.append(
                f"category:{category.strip()}"
            )

    return tuple(signals)


def _url_signals(
    candidate: NewsCandidate,
) -> tuple[str, ...]:
    """Находит кино-сигналы в пути URL."""

    path = urlsplit(
        candidate.source_url
    ).path.casefold()

    return tuple(
        f"url:{url_part}"
        for url_part in _STRONG_URL_PARTS
        if url_part in path
    )


def _text_signals(
    candidate: NewsCandidate,
) -> tuple[str, ...]:
    """Находит сильные кино-сигналы в тексте."""

    searchable_text = " ".join(
        part
        for part in (
            candidate.title,
            candidate.summary or "",
        )
        if part
    )

    matches = {
        match.group(0).casefold()
        for match in _STRONG_TEXT_PATTERN.finditer(
            searchable_text
        )
    }

    return tuple(
        f"text:{match}"
        for match in sorted(matches)
    )


def _industry_business_signals(
    candidate: NewsCandidate,
) -> tuple[str, ...]:
    """
    Распознаёт киноиндустриальную бизнес-новость.

    Требуется одновременно название киностудии
    и термин сделки, регулирования или производства.
    """

    searchable_text = _normalize_text(
        " ".join(
            part
            for part in (
                candidate.title,
                candidate.summary or "",
            )
            if part
        )
    )

    matched_entities = tuple(
        entity
        for entity in _FILM_INDUSTRY_ENTITIES
        if entity in searchable_text
    )

    matched_actions = tuple(
        action
        for action in _FILM_INDUSTRY_ACTIONS
        if action in searchable_text
    )

    if not matched_entities or not matched_actions:
        return ()

    return (
        "industry_entity:"
        + ",".join(matched_entities),
        "industry_action:"
        + ",".join(matched_actions),
    )


def evaluate_cinema_relevance(
    candidate: NewsCandidate,
) -> CinemaRelevanceDecision:
    """Оценивает одну публикацию без модели."""

    editorial_signals = (
        _editorial_exclusion_signals(
            candidate
        )
    )

    if editorial_signals:
        return CinemaRelevanceDecision(
            news_id=candidate.news_id,
            source_code=candidate.source_code,
            is_relevant=False,
            signals=editorial_signals,
            reason=(
                "Исключён явный редакционный "
                "формат без самостоятельного "
                "новостного инфоповода."
            ),
        )

    if not candidate.requires_cinema_relevance_filter:
        return CinemaRelevanceDecision(
            news_id=candidate.news_id,
            source_code=candidate.source_code,
            is_relevant=True,
            signals=("source_scope:curated",),
            reason=(
                "Источник уже ограничен "
                "кинематографической тематикой."
            ),
        )

    signals = (
        _category_signals(candidate)
        + _url_signals(candidate)
        + _text_signals(candidate)
        + _industry_business_signals(candidate)
    )

    unique_signals = tuple(
        dict.fromkeys(signals)
    )

    if unique_signals:
        return CinemaRelevanceDecision(
            news_id=candidate.news_id,
            source_code=candidate.source_code,
            is_relevant=True,
            signals=unique_signals,
            reason=(
                "Обнаружен как минимум один "
                "сильный сигнал связи с "
                "кинематографом."
            ),
        )

    return CinemaRelevanceDecision(
        news_id=candidate.news_id,
        source_code=candidate.source_code,
        is_relevant=False,
        signals=(),
        reason=(
            "Сильные сигналы связи с "
            "кинематографом не обнаружены."
        ),
    )


def filter_cinema_relevance(
    selection: CandidateSelectionResult,
) -> CinemaRelevanceFilterResult:
    """Фильтрует выборку до OpenAI."""

    decisions = tuple(
        evaluate_cinema_relevance(candidate)
        for candidate in selection.candidates
    )

    included_news_ids = {
        decision.news_id
        for decision in decisions
        if decision.is_relevant
    }

    filtered_selection = CandidateSelectionResult(
        window_start=selection.window_start,
        window_end=selection.window_end,
        window_hours=selection.window_hours,
        candidates=tuple(
            candidate
            for candidate in selection.candidates
            if candidate.news_id in included_news_ids
        ),
    )

    return CinemaRelevanceFilterResult(
        selection=filtered_selection,
        decisions=decisions,
    )
