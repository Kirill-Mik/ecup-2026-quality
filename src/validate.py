"""Валидация submit.csv перед записью: битая строка формата = ноль баллов."""
import re
from typing import List, Tuple

import pandas as pd

_RESULT_RE = re.compile(
    r"^<комментарий>(?P<comment>.{50,300}?)<вердикт>(?P<verdict>бан|не бан)$",
    re.DOTALL,
)


def validate_result_value(value: str) -> Tuple[bool, str]:
    if not isinstance(value, str):
        return False, "result не строка"
    m = _RESULT_RE.match(value)
    if not m:
        return False, "не совпал шаблон <комментарий>...<вердикт>..."
    comment = m.group("comment")
    if "<вердикт>" in comment or "<комментарий>" in comment:
        return False, "служебный тег внутри комментария"
    if not (50 <= len(comment) <= 300):
        return False, f"длина комментария {len(comment)} вне 50-300"
    return True, ""


def validate_submission(df: pd.DataFrame, expected_ids=None) -> List[str]:
    """Список ошибок; пустой список = файл валиден."""
    errors = []
    if list(df.columns) != ["id", "result"]:
        errors.append(f"колонки {list(df.columns)}, ожидались ['id', 'result']")
    if df["id"].duplicated().any():
        errors.append("дубликаты id")
    if df["result"].isna().any():
        errors.append("пустые result")
    if expected_ids is not None:
        missing = set(expected_ids) - set(df["id"])
        extra = set(df["id"]) - set(expected_ids)
        if missing:
            errors.append(f"нет ответов для {len(missing)} id")
        if extra:
            errors.append(f"{len(extra)} лишних id")
    for pos, value in enumerate(df["result"].tolist()):
        ok, why = validate_result_value(value)
        if not ok:
            errors.append(f"строка {pos}: {why}")
            if len(errors) > 20:
                errors.append("...дальше не проверял")
                break
    return errors
