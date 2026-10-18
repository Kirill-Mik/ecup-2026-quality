"""Комментарии к вердиктам: только проверенные шаблоны.

LLM-генерация «красивых» обоснований сознательно вырезана: метрика от
комментария не зависит (оценивается только вердикт), а генерация тянула
модель из /shared_models — шаг, который невозможно отрепетировать дома.
Каждый шаблон гарантированно даёт 50-300 символов после подстановки.
"""
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    CAT_BAD, CAT_FLAM, MAX_COMMENT_LEN, MIN_COMMENT_LEN, RESULT_TEMPLATE,
    VERDICT_BAN, VERDICT_OK,
)

_TEMPLATES = {
    (CAT_BAD, 0): (
        "Товар имеет признаки биологически активной добавки согласно регламенту "
        "категории: маркировки и состав указывают на БАД, карточка подлежит "
        "блокировке до подтверждения документов."
    ),
    (CAT_BAD, 1): (
        "Признаков нарушения регламента категории БАД не обнаружено: товар не "
        "содержит маркировок биологически активной добавки либо относится к "
        "спортивному питанию, ограничения не применяются."
    ),
    (CAT_FLAM, 0): (
        "Товар имеет признаки легковоспламеняющегося согласно регламенту: он "
        "содержит горючие вещества или является источником открытого огня, "
        "карточка подлежит блокировке."
    ),
    (CAT_FLAM, 1): (
        "Признаков нарушения регламента легковоспламеняющихся товаров не "
        "обнаружено: товар не содержит горючих веществ и не является источником "
        "воспламенения, ограничения не применяются."
    ),
}


def _sanitize(comment: str) -> str:
    """Комментарий не должен ломать формат: убираем угловые скобки и переносы."""
    c = str(comment).replace("<", "(").replace(">", ")")
    c = " ".join(c.split())
    return c.strip()


def _fit_length(comment: str, category: str, pred: int) -> str:
    c = _sanitize(comment)
    if len(c) < MIN_COMMENT_LEN or len(c) > MAX_COMMENT_LEN:
        fallback = _TEMPLATES[(category, int(pred))]
        if len(c) > MAX_COMMENT_LEN:
            cut = c.rfind(" ", MIN_COMMENT_LEN, MAX_COMMENT_LEN)
            c = c[:cut] if cut != -1 else c[:MAX_COMMENT_LEN]
        if len(c) < MIN_COMMENT_LEN:
            c = fallback
    return c


def template_comment(category: str, pred: int) -> str:
    return _TEMPLATES[(category, int(pred))]


def template_comments(df: pd.DataFrame, preds: np.ndarray) -> List[str]:
    cats = df["category"].values
    return [template_comment(cats[i], int(preds[i])) for i in range(len(df))]


def build_results(df: pd.DataFrame, preds: np.ndarray, comments: List[str]) -> List[str]:
    out = []
    cats = df["category"].values
    for i in range(len(df)):
        verdict = VERDICT_OK if preds[i] == 1 else VERDICT_BAN
        comment = _fit_length(comments[i], cats[i], int(preds[i]))
        out.append(RESULT_TEMPLATE.format(comment=comment, verdict=verdict))
    return out
