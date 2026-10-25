"""Различаем замену инфоповода и правку формулировок без платного классификатора."""
import re


class ReplacementPositionRequired(ValueError):
    pass


def replacement_positions(comment: str) -> tuple[int, ...]:
    text = comment.lower().replace('ё','е')
    replace = re.search(r'\b(замен\w*|убер\w*|исключ\w*)\b', text)
    duplicate = re.search(r'\b(дубл\w*|повтор\w*|вчерашн\w*)\b|уже\s+был[аои]?', text)
    if re.search(r'\bне\s+(замен|убер|исключ)', text):
        return ()
    if replace and re.search(r'\b(?:замен\w*|убер\w*|исключ\w*)\s+(?:формулиров\w*|слов\w*|фраз\w*|дат[ауые]|названи\w*|имя|текст)', text):
        return ()
    if re.search(r'повтор\w*\s+(?:слов|фраз)', text):
        return ()
    if not replace and not duplicate:
        return ()
    if re.search(r'\bвсе\s+(?:три\s+)?новост', text):
        return (1,2,3)
    positions = set()
    for position, word in ((1,'перв'),(2,'втор'),(3,'трет')):
        if re.search(r'\b'+word+r'\w*\s+новост|\bновост\w*\s+'+word, text):
            positions.add(position)
    # Число должно быть связано с новостью/заменой, а не ценой или датой.
    for match in re.finditer(r'(?:новост\w*\s*(?:№|номер)?\s*|(?:замен\w*|убер\w*|исключ\w*)\s*(?:новост\w*\s*)?)([123])\b', text):
        positions.add(int(match.group(1)))
    for match in re.finditer(r'\b([123])\s*[-–]?\s*(?:ю|ую|я|ая|й)\s+новост', text):
        positions.add(int(match.group(1)))
    if not positions:
        raise ReplacementPositionRequired('Для замены укажите номер новости: например, «Замени третью новость». Текст пока не изменён.')
    return tuple(sorted(positions))
