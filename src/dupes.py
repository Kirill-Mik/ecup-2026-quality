"""Мемоизация дубликатов: карточки маркетплейса повторяются.

Замер на трейне: 41.6% строк — дубликаты по (name+desc), согласие меток
внутри групп 96.7%; по одному только name кросс-фолдовое покрытие 8.5%
с точностью 85.8%. Если тест сэмплирован из того же пула (а это так для
public/private сплитов), точное совпадение с трейном — почти бесплатные
правильные ответы.

Два источника для бленда:
  dup_exact — хэш нормализованных name+description (вес задаётся вручную,
              OOF его измерить не может: фолды нарезаны как раз так, чтобы
              дубликаты не пересекали фолды);
  dup_name  — хэш только name (слабее, но OOF-измерим).
"""
import hashlib
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.rules import normalize_text

ARTIFACT = ARTIFACTS_DIR / "dupes.joblib"
ALPHA = 0.5                      # сглаживание Лапласа
P_CLIP_EXACT = (0.02, 0.98)
P_CLIP_NAME = (0.10, 0.90)


def exact_key(name: str, desc: str) -> str:
    s = normalize_text(name)[:200] + "|" + normalize_text(desc)[:300]
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def name_key(name: str) -> str:
    return hashlib.md5(normalize_text(name)[:200].encode("utf-8")).hexdigest()


def build_maps(df: pd.DataFrame) -> dict:
    """{exact: {hash: (p, n)}, name: {hash: (p, n)}} из трейна с метками."""
    maps = {"exact": {}, "name": {}}
    for kind, key_fn in (("exact", lambda r: exact_key(r["name"], r.get("description", ""))),
                         ("name", lambda r: name_key(r["name"]))):
        keys = df.apply(key_fn, axis=1)
        grouped = pd.DataFrame({"k": keys, "y": df["label"].values}).groupby("k")["y"]
        agg = grouped.agg(["sum", "count"])
        p = (agg["sum"] + ALPHA) / (agg["count"] + 2 * ALPHA)
        maps[kind] = {k: (float(pp), int(n))
                      for k, pp, n in zip(agg.index, p, agg["count"])}
    return maps


def save_maps(maps: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(maps, ARTIFACT)


def predict(df: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float]]:
    """(dup_exact, dup_name): {index: P(ok)} по хэш-совпадению с трейном."""
    if not ARTIFACT.exists():
        return {}, {}
    maps = joblib.load(ARTIFACT)
    exact, name = {}, {}
    names = df["name"].fillna("").astype(str)
    descs = df["description"].fillna("").astype(str) if "description" in df else pd.Series("", index=df.index)
    for idx, n, d in zip(df.index, names, descs):
        hit = maps["exact"].get(exact_key(n, d))
        if hit is not None:
            exact[int(idx)] = float(np.clip(hit[0], *P_CLIP_EXACT))
        hit = maps["name"].get(name_key(n))
        if hit is not None:
            name[int(idx)] = float(np.clip(hit[0], *P_CLIP_NAME))
    return exact, name
