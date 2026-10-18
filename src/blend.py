"""Смешивание источников по профилям деградации + страховка порогов.

Схема artifacts/blend.json (пишет train/tune_blend.py):
  {"source_order": [...],
   "dup_exact_weight": 3.0,
   "profiles": [
      {"name": "full", "requires": ["knn", "e5"],
       "weights":    {"БАД": {"text_t3": 1.0, ...}, ...},
       "thresholds": {"БАД": 0.61, ...},
       "pos_rate":   {"БАД": {"target": 0.72, "lo": 0.47, "hi": 0.92}, ...},
       "oof_a": 0.91},
      {"name": "no_e5", "requires": [], ...}]}

Три жёстких урока сабмитов №1/№2, зашитых в этот модуль:
 1. Вес НЕуказанного источника = 0 — ровно как в тюнере. В старом коде
    тюнер считал с дефолтом 0, а рантайм с дефолтом 1: дома оценивалась
    одна модель, на сервер ехала другая.
 2. Профиль выбирается по фактически посчитанным источникам: если тяжёлая
    голова не доехала, включаются веса И ПОРОГИ, подобранные на OOF именно
    без неё, — а не пороги полного состава.
 3. rate_guard: если доля «не бан» в категории вышла за коридор OOF,
    порог заменяется квантильным. Это страховка от «ЛВЖ схлопнулась в
    всегда-бан», которую не поймать без серверных логов.

dup_exact — особый источник (в OOF не существует: фолды нарезаны так, что
точные дубликаты не пересекают фолды): подмешивается ПОВЕРХ результата
профиля с фиксированным весом.
"""
import json
import math
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM, DEFAULT_THRESHOLDS

BLEND_FILE = "blend.json"
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_FALLBACK_CFG = {
    "source_order": [],
    "dup_exact_weight": 3.0,
    "profiles": [{
        "name": "fallback",
        "requires": [],
        "weights": {CAT_BAD: {}, CAT_FLAM: {}},
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "pos_rate": {},
        "oof_a": None,
    }],
}


def load_deploy() -> dict:
    """artifacts/blend.json; при любой проблеме — равновесный фолбэк."""
    path = ARTIFACTS_DIR / BLEND_FILE
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg.get("profiles"), "нет профилей"
        return cfg
    except Exception as e:  # noqa: BLE001
        print(f"[blend] blend.json не прочитался ({e}) — равновесный фолбэк",
              flush=True)
        return _FALLBACK_CFG


def select_profile(cfg: dict, present: Set[str]) -> dict:
    """Первый профиль, все requires которого фактически посчитаны.
    Последний профиль обязан иметь requires=[] — он и есть дно деградации."""
    for prof in cfg["profiles"]:
        if set(prof.get("requires", [])) <= present:
            return prof
    return cfg["profiles"][-1]


def blend_with_profile(
    df: pd.DataFrame,
    sources: Dict[str, Dict[int, float]],
    profile: dict,
    dup_exact_weight: float = 3.0,
) -> np.ndarray:
    """P(ok) на строку. Вес неуказанного источника = 0 (как в тюнере)."""
    dup_map = sources.get("dup_exact", {})
    base_sources = {k: v for k, v in sources.items() if k != "dup_exact"}
    cats = df["category"].values

    probs = np.empty(len(df), dtype=np.float64)
    n_orphan = 0
    for pos, idx in enumerate(df.index):
        w_cat = profile.get("weights", {}).get(cats[pos], {})
        num, den = 0.0, 0.0
        for name, mapping in base_sources.items():
            if idx not in mapping:
                continue
            w = float(w_cat.get(name, 0.0))
            if w <= 0:
                continue
            num += w * _logit(mapping[idx])
            den += w
        if den > 0:
            p = _sigmoid(num / den)
        else:
            # ни одного весового источника на строке: fallback-профиль без
            # весов ИЛИ все головы промахнулись мимо строки. Равные веса
            # по всему, что есть, — хуже тюненного, но лучше константы 0.5.
            row = [mapping[idx] for mapping in base_sources.values()
                   if idx in mapping]
            p = _sigmoid(float(np.mean([_logit(v) for v in row]))) if row else 0.5
            n_orphan += 1
        if idx in dup_map and dup_exact_weight > 0:
            p = _sigmoid((_logit(p) + dup_exact_weight * _logit(dup_map[idx]))
                         / (1.0 + dup_exact_weight))
        probs[pos] = p
    if n_orphan:
        print(f"[blend] строк без весовых источников: {n_orphan} "
              f"(профиль {profile.get('name')})", flush=True)
    return probs


def apply_thresholds(df: pd.DataFrame, probs_ok: np.ndarray,
                     thresholds: Optional[dict] = None) -> np.ndarray:
    """P(ok) -> pred (1 = «не бан», 0 = «бан») по порогу своей категории."""
    thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
    cats = df["category"].values
    preds = np.zeros(len(df), dtype=np.int64)
    for i in range(len(df)):
        thr = float(thresholds.get(cats[i], 0.5))
        preds[i] = int(probs_ok[i] >= thr)
    return preds


def rate_guard(df: pd.DataFrame, probs_ok: np.ndarray, preds: np.ndarray,
               profile: dict) -> Tuple[np.ndarray, list]:
    """Страховка от схлопывания категории.

    Если доля предсказанных «не бан» в категории вышла за коридор [lo, hi],
    наблюдавшийся на OOF, порог категории заменяется квантильным: банится
    ровно (1 - target) доля наименее уверенных строк. Квантильный порог
    устойчив к сдвигу распределения скоров — главному режиму отказа при
    потере/деградации источника на сервере.
    """
    notes = []
    preds = preds.copy()
    cats = df["category"].values
    for cat, band in (profile.get("pos_rate") or {}).items():
        m = cats == cat
        if not m.any():
            continue
        rate = float(preds[m].mean())
        lo, hi, target = band.get("lo"), band.get("hi"), band.get("target")
        if lo is None or hi is None or target is None:
            continue
        if lo <= rate <= hi:
            notes.append(f"{cat}: доля ok={rate:.3f} в коридоре [{lo:.3f}, {hi:.3f}]")
            continue
        qthr = float(np.quantile(probs_ok[m], 1.0 - float(target)))
        preds[m] = (probs_ok[m] >= qthr).astype(np.int64)
        new_rate = float(preds[m].mean())
        notes.append(
            f"{cat}: доля ok={rate:.3f} ВНЕ коридора [{lo:.3f}, {hi:.3f}] — "
            f"квантильный порог {qthr:.4f}, новая доля {new_rate:.3f}")
    return preds, notes
