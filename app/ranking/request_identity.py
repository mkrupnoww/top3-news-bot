"""Сравнение JSON-запросов без изменения исходных строк и checkpoint identity."""
import json


PREPARED_REQUEST_MISMATCH = (
    "Подготовленный event-level запрос не соответствует "
    "текущей выборке, модели или промпту."
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторяющийся ключ JSON')
        result[key] = value
    return result


def same_event_ranking_request(saved, expected) -> bool:
    """Порядок ключей/пробелы несущественны; модель, инструкции, значения и списки строги."""
    if saved.model != expected.model or saved.instructions != expected.instructions:
        return False
    if saved.input_text == expected.input_text:
        return True
    try:
        def canonical(text):
            payload = json.loads(text, object_pairs_hook=_unique_object)
            if not isinstance(payload, dict):
                raise ValueError('Ожидается объект JSON')
            return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return canonical(saved.input_text) == canonical(expected.input_text)
    except (ValueError, TypeError):
        return False
