"""Юнит-тесты рантайма сабмита: формат, бленд, профили, страховки.

  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.blend import (apply_thresholds, blend_with_profile, rate_guard,  # noqa: E402
                       select_profile)
from src.comments import build_results, template_comments  # noqa: E402
from src.config import CAT_BAD, CAT_FLAM, norm_category  # noqa: E402
from src.validate import validate_result_value, validate_submission  # noqa: E402


def _df(cats):
    return pd.DataFrame({"id": range(len(cats)), "category": cats})


# ------------------------------------------------------------------ формат
def test_templates_valid_for_all_cases():
    df = _df([CAT_BAD, CAT_BAD, CAT_FLAM, CAT_FLAM])
    preds = np.array([0, 1, 0, 1])
    results = build_results(df, preds, template_comments(df, preds))
    for r in results:
        ok, why = validate_result_value(r)
        assert ok, why


def test_validate_catches_broken_rows():
    assert not validate_result_value("<комментарий>коротко<вердикт>бан")[0]
    assert not validate_result_value("<комментарий>" + "х" * 100)[0]
    assert not validate_result_value("<комментарий>" + "х" * 100 + "<вердикт>ban")[0]
    assert validate_result_value("<комментарий>" + "х" * 100 + "<вердикт>не бан")[0]


def test_validate_submission_checks_ids():
    df = pd.DataFrame({"id": [1, 2],
                       "result": ["<комментарий>" + "х" * 60 + "<вердикт>бан"] * 2})
    assert validate_submission(df, expected_ids=[1, 2]) == []
    assert validate_submission(df, expected_ids=[1, 2, 3])


def test_norm_category():
    assert norm_category(" БАДы ") == CAT_BAD
    assert norm_category("бад") == CAT_BAD
    assert norm_category("Легковоспламеняющиеся товары") == CAT_FLAM


# ------------------------------------------------------------------ профили
_CFG = {
    "dup_exact_weight": 3.0,
    "profiles": [
        {"name": "full", "requires": ["knn", "e5"],
         "weights": {CAT_BAD: {"a": 1.0}, CAT_FLAM: {"a": 1.0}},
         "thresholds": {CAT_BAD: 0.5, CAT_FLAM: 0.5}},
        {"name": "no_e5", "requires": [],
         "weights": {CAT_BAD: {"a": 1.0}, CAT_FLAM: {"a": 1.0}},
         "thresholds": {CAT_BAD: 0.6, CAT_FLAM: 0.4}},
    ],
}


def test_profile_degrades_without_required_sources():
    assert select_profile(_CFG, {"a", "knn", "e5"})["name"] == "full"
    assert select_profile(_CFG, {"a", "knn"})["name"] == "no_e5"
    assert select_profile(_CFG, set())["name"] == "no_e5"


def test_unlisted_source_has_zero_weight():
    """Урок сабмита №2: тюнер считал вес неуказанного источника нулём,
    рантайм — единицей. Теперь оба считают нулём."""
    df = _df([CAT_BAD])
    prof = _CFG["profiles"][0]
    only_a = blend_with_profile(df, {"a": {0: 0.9}}, prof)
    with_noise = blend_with_profile(df, {"a": {0: 0.9}, "b": {0: 0.01}}, prof)
    assert only_a[0] == pytest.approx(with_noise[0])


def test_dup_exact_overlay():
    df = _df([CAT_BAD])
    prof = _CFG["profiles"][0]
    base = blend_with_profile(df, {"a": {0: 0.5}}, prof)
    boosted = blend_with_profile(df, {"a": {0: 0.5}, "dup_exact": {0: 0.98}},
                                 prof, dup_exact_weight=3.0)
    assert boosted[0] > base[0] + 0.2


# ------------------------------------------------------------------ страховки
def test_rate_guard_fixes_collapsed_category():
    """«ЛВЖ схлопнулась в всегда-бан» -> квантильный порог возвращает
    долю ok к OOF-целевой."""
    n = 200
    df = _df([CAT_FLAM] * n)
    probs = np.linspace(0.0, 0.4, n)          # все ниже порога 0.5
    preds = apply_thresholds(df, probs, {CAT_FLAM: 0.5})
    assert preds.sum() == 0
    prof = {"pos_rate": {CAT_FLAM: {"target": 0.05, "lo": 0.01, "hi": 0.15}}}
    fixed, notes = rate_guard(df, probs, preds, prof)
    assert 0.03 <= fixed.mean() <= 0.07
    assert notes and "ВНЕ коридора" in notes[0]


def test_rate_guard_keeps_healthy_predictions():
    n = 100
    df = _df([CAT_BAD] * n)
    probs = np.linspace(0, 1, n)
    preds = apply_thresholds(df, probs, {CAT_BAD: 0.3})
    prof = {"pos_rate": {CAT_BAD: {"target": 0.7, "lo": 0.45, "hi": 0.9}}}
    fixed, _ = rate_guard(df, probs, preds, prof)
    assert (fixed == preds).all()
